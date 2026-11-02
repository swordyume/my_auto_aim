// 断帧 / 打灭工况下的 CA 卡尔曼仿真（纯离线，不接相机/云台）
//
// 目的：按真实观测情况生成"无偏"的量测输入，喂给 singer 分支真实的 Target/EKF，
//       看断帧（最近装甲板被打灭 4~6 帧）会不会让 w / yaw / 瞄准点产生系统性偏差。
//
// 观测模型（可直接命令行调节）：
//   - 真值：4 块板绕中心自转，半径 0.285，间隔 90°，中心固定
//   - 平时：按入射角阈值只送 1~2 块板（相机不可能同时看到 4 块）
//   - 打灭：离相机最近的那块板（min |P|）灭 4~6 帧，期间只剩 0~1 块掠射板
//   - 量测：(az, pitch, dist, armor_yaw) = 真值 + 零均值高斯噪声，无任何偏置
//   - 某帧一块板都没有 → 该帧不调用 update()，只 predict()（和 tracker.cpp 一致）
//
// 输出：sim/logs/<tag>.rlog，用 ./host/replay.sh sim/logs/<tag>.rlog 查看曲线/图像/3D

#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/kalman_markers.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/target.hpp"
#include "tools/img_tools.hpp"
#include "tools/math_tools.hpp"
#include "tools/rdbg/session.hpp"
#include "tools/trajectory.hpp"
#include "tools/yaml.hpp"

namespace
{

constexpr double RAD2DEG = 180.0 / CV_PI;

struct Args
{
  std::string config = "configs/sentry.yaml";
  std::string preset = "dark";  // baseline | dark | dark_slow
  double w = 11.35;             // rad/s
  double dist = 2.666;          // |C|, m
  double az = 0.695;            // 中心方位角, rad
  double z = 0.0;               // 中心高度, m
  double radius = 0.285;        // 真值半径, m
  double init_radius = -1.0;    // <0 表示用真值半径；否则用配置值
  double fps = 122.0;
  double duration = 20.0;
  double warmup = 7.0;          // 打灭事件从 warmup 之后开始
  double stats_from = 2.5;      // 统计起点
  int seed = 1;
  int dark = 1;                 // 1 = 启用打灭
  double dark_period = 0.5;   // 打灭事件间隔（默认一块板转过来的时间）
  int dark_frames = 0;          // 0 = 10~12 随机
  double phi_detect = 70.0;     // 入射角检测阈值（度）
  double sigma_az = 0.005;      // rad
  double sigma_pitch = 0.003;   // rad
  double sigma_dist = 0.05;     // m
  double sigma_yaw = 0.14;      // rad
  double bias_yaw = 0.0;        // deg，掠射角下的 armor_yaw 偏置（复现输入偏置假设）
  double bias_az = 0.0;         // deg，掠射角下的方位角偏置
  double bias_angle = 60.0;     // deg，入射角超过该值时施加 bias
  double latency_ms = 0.0;      // 观测相对状态的时间滞后（模拟相机时间戳滞后）
  double bullet_speed = 22.0;
  bool image = true;
  int image_every = 4;          // 122/4 ≈ 30 fps
  std::string out = "sim/logs";
  std::string tag;
};

void print_usage(const char * argv0)
{
  fmt::print(
    "Usage: {} [options]\n"
    "  --config=PATH        yaml (default configs/sentry.yaml)\n"
    "  --preset=NAME        baseline | dark | dark_slow (default dark)\n"
    "  --w=11.35            目标角速度 rad/s\n"
    "  --dist=2.666         目标中心距离 m\n"
    "  --az=0.695           目标中心方位角 rad\n"
    "  --radius=0.285       真值半径 m\n"
    "  --init-radius=R      Target 初始半径（<0 用真值，默认）\n"
    "  --fps=122            仿真帧率\n"
    "  --duration=20        仿真时长 s\n"
    "  --seed=1             随机种子\n"
    "  --dark=1             1 启用打灭 / 0 基线\n"
    "  --dark-period=0.138  打灭事件间隔 s\n"
    "  --dark-frames=0      0 = 4~6 随机，>0 固定帧数\n"
    "  --phi-detect=70      入射角检测阈值（度）\n"
    "  --sigma-az=0.005     方位角噪声 rad\n"
    "  --sigma-pitch=0.003  俯仰角噪声 rad\n"
    "  --sigma-dist=0.05    距离噪声 m\n"
    "  --sigma-yaw=0.14     装甲板朝向噪声 rad\n"
    "  --bias-yaw=0         掠射角下的 armor_yaw 偏置 deg（负=观测偏左）\n"
    "  --bias-az=0          掠射角下的方位角偏置 deg\n"
    "  --bias-angle=60      入射角超过该值时施加 bias deg\n"
    "  --latency-ms=0       观测相对状态的时间滞后 ms\n"
    "  --bullet-speed=22    弹速 m/s\n"
    "  --image=1            1 写图像 / 0 不写\n"
    "  --image-every=4      每 N 帧写一张\n"
    "  --out=sim/logs       rlog 输出目录\n"
    "  --tag=NAME           rlog 文件名（默认 preset_s<seed>）\n",
    argv0);
}

Args parse_args(int argc, char ** argv)
{
  Args a;
  std::map<std::string, std::string> kv;
  for (int i = 1; i < argc; i++) {
    std::string s = argv[i];
    if (s == "-h" || s == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    }
    auto eq = s.find('=');
    if (eq == std::string::npos || s.rfind("--", 0) != 0) {
      fmt::print(stderr, "bad arg: {}\n", s);
      std::exit(2);
    }
    kv[s.substr(2, eq - 2)] = s.substr(eq + 1);
  }
  auto get = [&](const std::string & k) -> std::optional<std::string> {
    auto it = kv.find(k);
    if (it == kv.end()) return std::nullopt;
    return it->second;
  };
  auto num = [&](const std::string & k, double & dst) {
    if (auto v = get(k)) dst = std::stod(*v);
  };
  auto inum = [&](const std::string & k, int & dst) {
    if (auto v = get(k)) dst = std::stoi(*v);
  };

  if (auto v = get("config")) a.config = *v;
  if (auto v = get("preset")) a.preset = *v;
  num("w", a.w);
  num("dist", a.dist);
  num("az", a.az);
  num("z", a.z);
  num("radius", a.radius);
  num("init-radius", a.init_radius);
  num("fps", a.fps);
  num("duration", a.duration);
  num("warmup", a.warmup);
  num("stats-from", a.stats_from);
  inum("seed", a.seed);
  inum("dark", a.dark);
  num("dark-period", a.dark_period);
  inum("dark-frames", a.dark_frames);
  num("phi-detect", a.phi_detect);
  num("sigma-az", a.sigma_az);
  num("sigma-pitch", a.sigma_pitch);
  num("sigma-dist", a.sigma_dist);
  num("sigma-yaw", a.sigma_yaw);
  num("bias-yaw", a.bias_yaw);
  num("bias-az", a.bias_az);
  num("bias-angle", a.bias_angle);
  num("latency-ms", a.latency_ms);
  num("bullet-speed", a.bullet_speed);
  int image_i = a.image ? 1 : 0;
  inum("image", image_i);
  a.image = image_i != 0;
  inum("image-every", a.image_every);
  if (auto v = get("out")) a.out = *v;
  if (auto v = get("tag")) a.tag = *v;

  // preset 只改默认值，显式参数优先
  if (a.preset == "baseline") {
    a.dark = 0;
  } else if (a.preset == "dark") {
    a.dark = 1;
    if (!get("dark-period")) a.dark_period = 0.5;
  } else if (a.preset == "dark_slow") {
    a.dark = 1;
    if (!get("dark-period")) a.dark_period = 1.0;
  } else {
    fmt::print(stderr, "unknown preset: {}\n", a.preset);
    std::exit(2);
  }
  if (get("dark")) a.dark = std::stoi(*get("dark"));  // 显式覆盖 preset
  if (a.tag.empty()) a.tag = fmt::format("{}_s{}", a.preset, a.seed);
  return a;
}

struct Truth
{
  Eigen::Vector3d C;
  double r = 0.285;
  double l = 0.0;
  double h = 0.0;
  double yaw = 0.0;
  double w = 0.0;
  int n = 4;

  double angle(int i) const { return tools::limit_rad(yaw + i * 2.0 * CV_PI / n); }

  bool use_lh(int i) const { return n == 4 && (i == 1 || i == 3); }

  Eigen::Vector3d plate_xyz(int i) const { return plate_xyz_at(yaw, i); }

  Eigen::Vector3d plate_xyz_at(double y, int i) const
  {
    auto a = tools::limit_rad(y + i * 2.0 * CV_PI / n);
    auto rr = r + (use_lh(i) ? l : 0.0);
    return {C.x() - rr * std::cos(a), C.y() - rr * std::sin(a), C.z() + (use_lh(i) ? h : 0.0)};
  }

  double plate_yaw_at(double y, int i) const { return tools::limit_rad(y + i * 2.0 * CV_PI / n); }
};

// 入射角余弦：板法线（-n）与"板->相机"方向的夹角余弦
// P = C - r*n  =>  cos_inc = (n·C - r) / |P_xy|
double incidence_cos(const Truth & tr, int i)
{
  auto a = tr.angle(i);
  Eigen::Vector2d n(std::cos(a), std::sin(a));
  Eigen::Vector2d C(tr.C.x(), tr.C.y());
  Eigen::Vector3d P = tr.plate_xyz(i);
  double d = P.head<2>().norm();
  if (d < 1e-9) return 0.0;
  double rr = tr.r + (tr.use_lh(i) ? tr.l : 0.0);
  return (n.dot(C) - rr) / d;
}

int closest_plate(const Truth & tr)
{
  int best = 0;
  double best_d = 1e18;
  for (int i = 0; i < tr.n; i++) {
    double d = tr.plate_xyz(i).head<2>().norm();
    if (d < best_d) {
      best_d = d;
      best = i;
    }
  }
  return best;
}

// 与 Target::update() 内部一致的 id 匹配（取距离最近 3 块里角度误差最小的）
int match_id(const auto_aim::Target & target, const Eigen::Vector3d & obs_ypd, double obs_yaw)
{
  auto list = target.armor_xyza_list();
  std::vector<std::pair<double, int>> by_d;
  by_d.reserve(list.size());
  for (int i = 0; i < static_cast<int>(list.size()); i++) {
    auto ypd = tools::xyz2ypd(list[i].head<3>());
    by_d.push_back({ypd[2], i});
  }
  std::sort(by_d.begin(), by_d.end());
  int best = by_d.empty() ? -1 : by_d.front().second;
  double best_err = 1e18;
  for (int k = 0; k < std::min<int>(3, static_cast<int>(by_d.size())); k++) {
    int i = by_d[k].second;
    auto ypd = tools::xyz2ypd(list[i].head<3>());
    double err = std::abs(tools::limit_rad(obs_yaw - list[i][3])) +
                 std::abs(tools::limit_rad(obs_ypd[0] - ypd[0]));
    if (err < best_err) {
      best_err = err;
      best = i;
    }
  }
  return best;
}

auto_aim::Armor make_armor(
  const Eigen::Vector3d & xyz, const Eigen::Vector3d & ypd, double yaw, auto_aim::ArmorName name,
  auto_aim::ArmorType type)
{
  std::vector<cv::Point2f> pts{{0.f, 0.f}, {1.f, 0.f}, {1.f, 1.f}, {0.f, 1.f}};
  auto_aim::Armor a(auto_aim::Color::blue, name, 0.9f, cv::Rect(), pts);
  a.type = type;
  a.xyz_in_world = xyz;
  a.ypd_in_world = ypd;
  a.ypr_in_world = Eigen::Vector3d::Zero();
  a.ypr_in_world[0] = yaw;
  a.priority = auto_aim::ArmorPriority::first;
  return a;
}

struct Obs
{
  int plate = 0;
  Eigen::Vector3d xyz;
  Eigen::Vector3d ypd;
  double yaw = 0.0;
};

struct Rec
{
  double t = 0.0;
  bool has_target = false;
  double w = 0.0, yaw = 0.0, x = 0.0, y = 0.0, r = 0.0;
  int nobs = 0;
  bool dark = false;
  double aim_err = NAN, aim_err_red = NAN;
  int aim_state_id = -1, aim_true_id = -1;
  double in_az = NAN, in_pitch = NAN, in_dist = NAN, in_yaw = NAN;      // obs - truth
  double inn_az = NAN, inn_pitch = NAN, inn_dist = NAN, inn_yaw = NAN;  // obs - predict
  double obs1_az = NAN, obs1_pitch = NAN, obs1_dist = NAN, obs1_yaw = NAN;  // 第一块板的原始观测
  int obs1_plate = -1;
  double true1_az = NAN, true1_pitch = NAN, true1_dist = NAN, true1_yaw = NAN;  // 对应真值
};

std::vector<std::string> list_rlogs(const std::string & dir)
{
  std::vector<std::string> out;
  std::error_code ec;
  if (!std::filesystem::is_directory(dir, ec)) return out;
  for (auto & p : std::filesystem::directory_iterator(dir, ec)) {
    if (p.path().extension() == ".rlog") out.push_back(p.path().string());
  }
  return out;
}

double mean_of(const std::vector<double> & v)
{
  if (v.empty()) return NAN;
  double s = 0.0;
  for (double x : v) s += x;
  return s / v.size();
}

double pct_of(std::vector<double> v, double p)
{
  if (v.empty()) return NAN;
  std::sort(v.begin(), v.end());
  size_t k = std::min(v.size() - 1, static_cast<size_t>(p * (v.size() - 1)));
  return v[k];
}

double detrended_std(const std::vector<double> & t, const std::vector<double> & v)
{
  if (t.size() < 3) return NAN;
  double n = t.size(), st = 0, sv = 0, stt = 0, stv = 0;
  for (size_t i = 0; i < t.size(); i++) {
    st += t[i];
    sv += v[i];
    stt += t[i] * t[i];
    stv += t[i] * v[i];
  }
  double den = n * stt - st * st;
  if (std::abs(den) < 1e-12) return NAN;
  double k = (n * stv - st * sv) / den;
  double b = (sv - k * st) / n;
  double ss = 0;
  for (size_t i = 0; i < t.size(); i++) {
    double rr = v[i] - (k * t[i] + b);
    ss += rr * rr;
  }
  return std::sqrt(ss / n);
}

}  // namespace

int main(int argc, char ** argv)
{
  Args args = parse_args(argc, argv);
  const std::string cfg_path = tools::project_path(args.config);
  // 输出目录相对工程根解析，避免 CLion / 其他 CWD 下写到别处
  const std::string out_dir = tools::project_path(args.out);

  // ---------------- 配置 ----------------
  auto yaml = tools::load(cfg_path);
  auto ekf = yaml["ekf"];
  auto_aim::EkfParams ekf_params;
  ekf_params.q_xyz_acc_var = ekf["q_xyz_acc_var"].as<double>();
  ekf_params.q_yaw_acc_var = ekf["q_yaw_acc_var"].as<double>();
  ekf_params.r_yaw = ekf["r_yaw"].as<double>();
  ekf_params.r_pitch = ekf["r_pitch"].as<double>();
  ekf_params.r_dist_base = ekf["r_dist_base"].as<double>();

  auto node = ekf["normal"];
  int armor_num = node["armor_num"].as<int>();
  auto p0_vec = node["p0"].as<std::vector<double>>();
  Eigen::VectorXd p0 = Eigen::Map<Eigen::VectorXd>(p0_vec.data(), p0_vec.size());

  const double decision_speed = yaml["decision_speed"].as<double>();
  const double high_delay = yaml["high_speed_delay_time"].as<double>();
  const double low_delay = yaml["low_speed_delay_time"].as<double>();
  const int max_temp_lost = yaml["max_temp_lost_count"] ? yaml["max_temp_lost_count"].as<int>() : 15;

  const double init_radius = args.init_radius >= 0.0 ? args.init_radius : args.radius;

  // ---------------- 真值 / Solver / rlog ----------------
  Truth truth;
  truth.C = Eigen::Vector3d(args.dist * std::cos(args.az), args.dist * std::sin(args.az), args.z);
  truth.r = args.radius;
  truth.w = args.w;
  truth.n = armor_num;
  const double yaw0 = 0.37;
  truth.yaw = yaw0;

  auto_aim::Solver solver(cfg_path);
  {
    // 固定云台指向目标中心（仿真不模拟云台运动）
    Eigen::Vector3d ypr(args.az, 0.0, 0.0);
    solver.set_R_gimbal2world(Eigen::Quaterniond(tools::rotation_matrix(ypr)));
  }

  const auto armor_name = auto_aim::ArmorName::three;
  const auto armor_type = auto_aim::ArmorType::small;

  std::error_code ec;
  std::filesystem::create_directories(out_dir, ec);
  auto before = list_rlogs(out_dir);

  tools::rdbg::Session session;
  session.open(out_dir);

  const auto wall0 = std::chrono::system_clock::now().time_since_epoch();
  const uint64_t ts0 =
    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(wall0).count()) +
    1000000ULL;

  std::mt19937_64 rng(static_cast<uint64_t>(args.seed) * 0x9E3779B97F4A7C15ULL + 12345ULL);
  std::normal_distribution<double> nd(0.0, 1.0);

  const double dt = 1.0 / args.fps;
  const int frames = static_cast<int>(std::llround(args.duration * args.fps));
  const double cos_phi = std::cos(args.phi_detect / RAD2DEG);

  std::optional<auto_aim::Target> target;
  int lost_frames = 0;
  int dark_plate = -1;
  int dark_until = -1;
  int next_dark = static_cast<int>(std::llround(args.warmup * args.fps));
  const int dark_period_frames =
    std::max(1, static_cast<int>(std::llround(args.dark_period * args.fps)));

  std::vector<Rec> recs;
  recs.reserve(frames);

  int dark_events = 0;
  int obs_hist[5] = {0, 0, 0, 0, 0};

  for (int k = 0; k < frames; k++) {
    const double t = k * dt;
    const uint64_t ts = ts0 + static_cast<uint64_t>(std::llround(t * 1e9));

    truth.yaw = tools::limit_rad(yaw0 + truth.w * t);
    Truth obs_truth = truth;
    obs_truth.yaw =
      tools::limit_rad(yaw0 + truth.w * (t - args.latency_ms / 1000.0));

    // ---- 打灭事件调度：把"离相机最近"的那块板打灭 ----
    bool is_dark = false;
    if (args.dark && k >= next_dark && k >= dark_until) {
      dark_plate = closest_plate(obs_truth);
      int nf = args.dark_frames > 0 ? args.dark_frames : (10 + static_cast<int>(rng() % 3));
      dark_until = k + nf;
      next_dark = k + dark_period_frames;
      dark_events++;
    }
    if (k < dark_until) is_dark = true;

    // ---- 生成观测（无偏） ----
    std::vector<Obs> obs;
    for (int i = 0; i < truth.n; i++) {
      if (is_dark && i == dark_plate) continue;
      double ci = incidence_cos(obs_truth, i);
      if (ci < cos_phi) continue;  // 掠射角检测不到

      Eigen::Vector3d xyz = obs_truth.plate_xyz(i);
      Eigen::Vector3d ypd = tools::xyz2ypd(xyz);
      Obs o;
      o.plate = i;
      o.ypd = ypd;
      o.yaw = obs_truth.plate_yaw_at(obs_truth.yaw, i);
      o.ypd[0] += nd(rng) * args.sigma_az;
      o.ypd[1] += nd(rng) * args.sigma_pitch;
      o.ypd[2] += nd(rng) * args.sigma_dist;
      o.yaw = tools::limit_rad(o.yaw + nd(rng) * args.sigma_yaw);
      if (args.bias_yaw != 0.0 || args.bias_az != 0.0) {
        double inc_deg = std::acos(std::clamp(ci, -1.0, 1.0)) * RAD2DEG;
        if (inc_deg > args.bias_angle) {
          o.yaw = tools::limit_rad(o.yaw + args.bias_yaw / RAD2DEG);
          o.ypd[0] += args.bias_az / RAD2DEG;
          o.xyz = tools::ypd2xyz(o.ypd);
        }
      }
      o.xyz = tools::ypd2xyz(o.ypd);
      obs.push_back(o);
    }
    obs_hist[std::min<int>(4, obs.size())]++;

    Rec r;
    r.t = t;
    r.nobs = static_cast<int>(obs.size());
    r.dark = is_dark;

    // ---- 驱动真实 Target/EKF（和 tracker.cpp 的 update_target 一致） ----
    if (!target) {
      if (!obs.empty()) {
        auto a = make_armor(obs[0].xyz, obs[0].ypd, obs[0].yaw, armor_name, armor_type);
        target.emplace(
          a, std::chrono::steady_clock::now(), init_radius, armor_num, p0, ekf_params);
        lost_frames = 0;
      }
    } else {
      target->predict(dt);

      if (obs.empty()) {
        lost_frames++;
        if (lost_frames > max_temp_lost) {
          target.reset();
          lost_frames = 0;
        }
      } else {
        lost_frames = 0;
        for (size_t oi = 0; oi < obs.size(); oi++) {
          const auto & o = obs[oi];
          // 记录"这一帧第一个观测"的输入误差和新息（新息必须在 update 之前算）
          if (oi == 0) {
            int id = match_id(*target, o.ypd, o.yaw);
            auto list = target->armor_xyza_list();
            if (id >= 0) {
              auto ypd_pred = tools::xyz2ypd(list[id].head<3>());
              r.inn_az = tools::limit_rad(o.ypd[0] - ypd_pred[0]);
              r.inn_pitch = tools::limit_rad(o.ypd[1] - ypd_pred[1]);
              r.inn_dist = o.ypd[2] - ypd_pred[2];
              r.inn_yaw = tools::limit_rad(o.yaw - list[id][3]);
            }
            auto ypd_true = tools::xyz2ypd(obs_truth.plate_xyz(o.plate));
            r.obs1_plate = o.plate;
            r.obs1_az = o.ypd[0] * RAD2DEG;
            r.obs1_pitch = o.ypd[1] * RAD2DEG;
            r.obs1_dist = o.ypd[2];
            r.obs1_yaw = o.yaw * RAD2DEG;
            r.true1_az = ypd_true[0] * RAD2DEG;
            r.true1_pitch = ypd_true[1] * RAD2DEG;
            r.true1_dist = ypd_true[2];
            r.true1_yaw = obs_truth.plate_yaw_at(obs_truth.yaw, o.plate) * RAD2DEG;
            r.in_az = tools::limit_rad(o.ypd[0] - ypd_true[0]);
            r.in_pitch = tools::limit_rad(o.ypd[1] - ypd_true[1]);
            r.in_dist = o.ypd[2] - ypd_true[2];
            r.in_yaw = tools::limit_rad(o.yaw - obs_truth.plate_yaw_at(obs_truth.yaw, o.plate));
          }
          auto a = make_armor(o.xyz, o.ypd, o.yaw, armor_name, armor_type);
          target->update(a);
        }
      }
    }

    Eigen::Vector3d aim_pred = Eigen::Vector3d::Zero();
    double aim_yaw = 0.0;
    if (target) {
      const auto & x = target->ekf_x();
      r.has_target = true;
      r.w = x[10];
      r.yaw = x[9];
      r.x = x[0];
      r.y = x[3];
      r.r = x[11];

      // ---- 瞄准点误差：和 planner 一致（delay + fly_time），取最近板 ----
      auto list_now = target->armor_xyza_list();
      double min_dist = 1e18;
      Eigen::Vector3d aim_xyz = Eigen::Vector3d::Zero();
      for (auto & xyza : list_now) {
        double d = xyza.head<2>().norm();
        if (d < min_dist) {
          min_dist = d;
          aim_xyz = xyza.head<3>();
        }
      }
      const double delay = std::abs(x[10]) > decision_speed ? high_delay : low_delay;
      const double fly = tools::Trajectory(args.bullet_speed, min_dist, aim_xyz.z()).fly_time;
      const double T = delay + fly;

      auto tgt_fly = *target;
      tgt_fly.predict(T);
      auto list_T = tgt_fly.armor_xyza_list();
      double best_d = 1e18;
      int best_i = -1;
      for (int i = 0; i < static_cast<int>(list_T.size()); i++) {
        double d = list_T[i].head<2>().norm();
        if (d < best_d) {
          best_d = d;
          best_i = i;
        }
      }
      if (best_i >= 0) {
        aim_pred = list_T[best_i].head<3>();
        aim_yaw = list_T[best_i][3];
        r.aim_state_id = best_i;

        // 真值里离预测点最近的那块板 = 实际瞄的那块
        const double yf = tools::limit_rad(truth.yaw + truth.w * T);
        double best_td = 1e18;
        int best_ti = -1;
        for (int i = 0; i < truth.n; i++) {
          double d = (truth.plate_xyz_at(yf, i) - aim_pred).norm();
          if (d < best_td) {
            best_td = d;
            best_ti = i;
          }
        }
        r.aim_true_id = best_ti;
        r.aim_err = best_td;

        // 红框对应的外推（fly-time 之后 0.50 s）
        auto tgt_red = *target;
        tgt_red.predict(T + 0.50);
        auto list_R = tgt_red.armor_xyza_list();
        double bd = 1e18;
        int bi = -1;
        for (int i = 0; i < static_cast<int>(list_R.size()); i++) {
          double d = list_R[i].head<2>().norm();
          if (d < bd) {
            bd = d;
            bi = i;
          }
        }
        if (bi >= 0) {
          const double yf2 = tools::limit_rad(truth.yaw + truth.w * (T + 0.50));
          double td = 1e18;
          for (int i = 0; i < truth.n; i++) {
            td = std::min(td, (truth.plate_xyz_at(yf2, i) - list_R[bi].head<3>()).norm());
          }
          r.aim_err_red = td;
        }
      }
    }

    recs.push_back(r);

    // ---- 写 plot ----
    {
      nlohmann::json j;
      j["t"] = t;
      j["n_obs"] = r.nobs;
      j["dark"] = r.dark ? 1 : 0;
      j["has_target"] = r.has_target ? 1 : 0;
      j["w"] = r.w;
      j["truth_w"] = truth.w;
      j["yaw"] = r.yaw;
      j["truth_yaw"] = truth.yaw;
      j["target_x"] = r.x;
      j["target_y"] = r.y;
      j["truth_x"] = truth.C.x();
      j["truth_y"] = truth.C.y();
      j["r"] = r.r;
      j["truth_r"] = truth.r;
      if (!std::isnan(r.aim_err)) j["aim_err_cm"] = r.aim_err * 100.0;
      if (!std::isnan(r.aim_err_red)) j["aim_err_red_cm"] = r.aim_err_red * 100.0;
      if (r.aim_state_id >= 0) j["aim_state_id"] = r.aim_state_id;
      if (r.aim_true_id >= 0) j["aim_true_id"] = r.aim_true_id;
      if (!std::isnan(r.in_az)) {
        j["in_az_deg"] = r.in_az * RAD2DEG;
        j["in_pitch_deg"] = r.in_pitch * RAD2DEG;
        j["in_dist_cm"] = r.in_dist * 100.0;
        j["in_yaw_deg"] = r.in_yaw * RAD2DEG;
        j["inn_az_deg"] = r.inn_az * RAD2DEG;
        j["inn_pitch_deg"] = r.inn_pitch * RAD2DEG;
        j["inn_dist_cm"] = r.inn_dist * 100.0;
        j["inn_yaw_deg"] = r.inn_yaw * RAD2DEG;
        j["obs1_az_deg"] = r.obs1_az;
        j["obs1_pitch_deg"] = r.obs1_pitch;
        j["obs1_dist_m"] = r.obs1_dist;
        j["obs1_yaw_deg"] = r.obs1_yaw;
        j["obs1_plate"] = r.obs1_plate;
        j["true1_az_deg"] = r.true1_az;
        j["true1_pitch_deg"] = r.true1_pitch;
        j["true1_dist_m"] = r.true1_dist;
        j["true1_yaw_deg"] = r.true1_yaw;
      }
      if (target) j["markers"] = tools::viz::kalman_markers(*target);
      session.write_json(ts, j.dump());
    }

    // ---- 写图像（可选） ----
    if (args.image && (k % std::max(1, args.image_every) == 0)) {
      cv::Mat img(1080, 1440, CV_8UC3, cv::Scalar(18, 18, 26));
      auto plate_center = [](const std::vector<cv::Point2f> & p) {
        cv::Point2f c(0, 0);
        for (auto & q : p) c += q;
        return c * (1.0f / static_cast<float>(p.size()));
      };
      // 真值 4 块板：蓝色填充 + 蓝边 + 蓝点
      std::vector<cv::Point2f> tcenters;
      for (int i = 0; i < truth.n; i++) {
        auto pts = solver.reproject_armor(
          truth.plate_xyz(i), truth.plate_yaw_at(truth.yaw, i), armor_type, armor_name);
        std::vector<std::vector<cv::Point>> poly(1);
        for (auto & p : pts) poly[0].push_back(cv::Point(cvRound(p.x), cvRound(p.y)));
        cv::fillPoly(img, poly, {80, 40, 0});
        tools::draw_points(img, pts, {255, 120, 0}, 2);
        auto c = plate_center(pts);
        tcenters.push_back(c);
        tools::draw_point(img, cv::Point(cvRound(c.x), cvRound(c.y)), {255, 120, 0}, 4);
      }
      // 估计的 4 块板：绿边 + 绿点 + 黄线指示估计->最近的真实板
      if (target) {
        auto est = target->armor_xyza_list();
        for (int i = 0; i < static_cast<int>(est.size()); i++) {
          auto pts =
            solver.reproject_armor(est[i].head<3>(), est[i][3], armor_type, armor_name);
          auto c = plate_center(pts);
          tools::draw_points(img, pts, {0, 255, 0}, 1);
          tools::draw_point(img, cv::Point(cvRound(c.x), cvRound(c.y)), {0, 255, 0}, 2);
          // 连到最近的真实板中心（id 只是标签，真正的物理板看最近）
          const cv::Point2f * best = nullptr;
          float bd = 1e18f;
          for (auto & tc : tcenters) {
            float d = (tc - c).dot(tc - c);
            if (d < bd) {
              bd = d;
              best = &tc;
            }
          }
          if (best) cv::line(img, c, *best, {0, 255, 255}, 1);
        }
      }
      // 瞄准点：红
      if (!std::isnan(r.aim_err)) {
        auto pts = solver.reproject_armor(aim_pred, aim_yaw, armor_type, armor_name);
        auto c = plate_center(pts);
        tools::draw_point(img, cv::Point(cvRound(c.x), cvRound(c.y)), {0, 0, 255}, 5);
        for (auto & p : pts) tools::draw_point(img, p, {0, 0, 255}, 1);
      }

      // 以真实装甲板中心裁剪 640x480（等效 2.25x 放大），文字画在裁剪图上
      cv::Point2f cc(0, 0);
      for (auto & q : tcenters) cc += q;
      if (!tcenters.empty()) cc *= (1.0f / static_cast<float>(tcenters.size()));
      const int cw = 640, ch = 480;
      int x0 = std::clamp(cvRound(cc.x) - cw / 2, 0, img.cols - cw);
      int y0 = std::clamp(cvRound(cc.y) - ch / 2, 0, img.rows - ch);
      cv::Mat crop = img(cv::Rect(x0, y0, cw, ch));

      tools::draw_text(
        crop,
        fmt::format("t={:.2f}s  w={:.2f}  nobs={}  {}", t, r.w, r.nobs, r.dark ? "DARK" : ""),
        {12, 26}, {220, 230, 255}, 0.7, 2);
      if (!std::isnan(r.aim_err)) {
        tools::draw_text(
          crop,
          fmt::format(
            "aim_err={:.1f}cm   green=est  blue=truth  red=aim", r.aim_err * 100.0),
          {12, 52}, {150, 210, 255}, 0.55, 1);
      }

      std::vector<uint8_t> jpg;
      std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, 75};
      if (cv::imencode(".jpg", crop, jpg, params)) {
        session.write_image(ts, R"({"name":"reprojection"})", jpg);
      }
    }
  }

  session.close();

  // ---------------- 汇总 ----------------
  std::vector<double> tw, wv, ae, aer, inn_az, inn_pitch, inn_dist, inn_yaw, in_az, in_pitch,
    in_dist, in_yaw;
  int n_dark = 0, n_stats = 0;
  double w_first = 0, w_last = 0;
  int n_first = 0, n_last = 0;
  const double t_end = recs.empty() ? 0.0 : recs.back().t;
  for (auto & r : recs) {
    if (r.t < args.stats_from || !r.has_target) continue;
    n_stats++;
    tw.push_back(r.t);
    wv.push_back(r.w);
    if (r.t < args.stats_from + 2.0) {
      w_first += r.w;
      n_first++;
    }
    if (r.t > t_end - 2.0) {
      w_last += r.w;
      n_last++;
    }
    if (r.dark) n_dark++;
    if (!std::isnan(r.aim_err)) ae.push_back(r.aim_err);
    if (!std::isnan(r.aim_err_red)) aer.push_back(r.aim_err_red);
    if (!std::isnan(r.inn_az)) {
      inn_az.push_back(r.inn_az);
      inn_pitch.push_back(r.inn_pitch);
      inn_dist.push_back(r.inn_dist);
      inn_yaw.push_back(r.inn_yaw);
      in_az.push_back(r.in_az);
      in_pitch.push_back(r.in_pitch);
      in_dist.push_back(r.in_dist);
      in_yaw.push_back(r.in_yaw);
    }
  }

  const double w_mean = mean_of(wv);
  double w_std = NAN;
  if (!wv.empty()) {
    double s = 0;
    for (double x : wv) s += (x - w_mean) * (x - w_mean);
    w_std = std::sqrt(s / wv.size());
  }
  const double w_drift = (n_first && n_last) ? (w_first / n_first - w_last / n_last) : NAN;
  const double w_detr = detrended_std(tw, wv);
  int hits = 0;
  for (double e : ae) hits += (e < 0.0675) ? 1 : 0;

  fmt::print(
    "\n================ sim summary ================\n"
    "config        : {}\n"
    "preset        : {}   seed={}\n"
    "truth         : w={:.3f} rad/s  r={:.3f} m  |C|={:.3f} m\n"
    "dark          : {}  period={:.3f}s  frames={}  events={}  dark_rate={:.1f}%\n"
    "obs count     : 0:{} 1:{} 2:{} 3:{} 4:{}\n"
    "stats window  : [{:.1f}, {:.1f}] s  n={}\n"
    "w             : mean={:.4f} std={:.4f} detrended_std={:.4f}\n"
    "w drift       : first2s-last2s = {:+.4f} rad/s\n"
    "aim_err_fly   : mean={:.2f}cm p95={:.2f}cm max={:.2f}cm  hit(<6.75cm)={:.1f}%\n"
    "aim_err_red   : mean={:.2f}cm p95={:.2f}cm max={:.2f}cm\n"
    "input bias    : az={:+.4f} pitch={:+.4f} dist={:+.4f}cm yaw={:+.4f} deg\n"
    "innovation    : az={:+.4f} pitch={:+.4f} dist={:+.4f}cm yaw={:+.4f} deg   (n={})\n"
    "                                  (az/pitch/yaw 单位 deg, dist 单位 cm)\n",
    cfg_path, args.preset, args.seed, args.w, args.radius, args.dist, args.dark ? "on" : "off",
    args.dark_period, args.dark_frames > 0 ? std::to_string(args.dark_frames) : std::string("4~6"),
    dark_events, args.dark ? 100.0 * n_dark / std::max(1, n_stats) : 0.0, obs_hist[0], obs_hist[1],
    obs_hist[2], obs_hist[3], obs_hist[4], args.stats_from, t_end, n_stats, w_mean, w_std, w_detr,
    w_drift, mean_of(ae) * 100, pct_of(ae, 0.95) * 100, pct_of(ae, 1.0) * 100,
    ae.empty() ? 0.0 : 100.0 * hits / ae.size(), mean_of(aer) * 100, pct_of(aer, 0.95) * 100,
    pct_of(aer, 1.0) * 100, mean_of(in_az) * RAD2DEG, mean_of(in_pitch) * RAD2DEG,
    mean_of(in_dist) * 100, mean_of(in_yaw) * RAD2DEG, mean_of(inn_az) * RAD2DEG,
    mean_of(inn_pitch) * RAD2DEG, mean_of(inn_dist) * 100, mean_of(inn_yaw) * RAD2DEG,
    inn_az.size());

  // 重命名 rlog 为可读名字
  auto after = list_rlogs(out_dir);
  std::set<std::string> before_set(before.begin(), before.end());
  std::string newest;
  std::filesystem::file_time_type newest_t{};
  for (auto & f : after) {
    if (before_set.count(f)) continue;
    auto ft = std::filesystem::last_write_time(f, ec);
    if (newest.empty() || ft > newest_t) {
      newest = f;
      newest_t = ft;
    }
  }
  std::string final_path;
  if (!newest.empty()) {
    final_path = (std::filesystem::path(out_dir) / (args.tag + ".rlog")).string();
    std::filesystem::rename(newest, final_path, ec);
    if (ec) {
      final_path = newest;
      fmt::print(stderr, "rename failed: {}\n", ec.message());
    }
  }
  fmt::print(
    "rlog          : {}\nview          : ./host/replay.sh {}\n",
    final_path.empty() ? "(none)" : final_path,
    final_path.empty() ? "sim/logs/xxx.rlog" : final_path);
  fmt::print("=============================================\n");

  return 0;
}
