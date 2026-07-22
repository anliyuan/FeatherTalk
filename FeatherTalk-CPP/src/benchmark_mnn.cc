#include <MNN/ErrorCode.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kFaceSize = 160;

struct Args {
  std::string feather_model;
  std::string unet_model;
  std::string wenet_model;
  std::string backend = "all";
  std::string precision = "high";
  int iterations = 300;
  int warmup = 50;
  int threads = 4;
};

void Usage() {
  std::cout
      << "Benchmark FeatherTalk MNN models (C++ / MNN)\n\n"
      << "Required:\n"
      << "  --feather-model PATH\n"
      << "  --unet-model PATH\n\n"
      << "Optional:\n"
      << "  --wenet-model PATH    Original Wenet encoder MNN model for comparison\n"
      << "  --backend all|cpu|metal|opencl  Backends to test (default: all)\n"
      << "  --precision high|normal|low    MNN compute precision (default: high)\n"
      << "  --iterations N                 Timed iterations (default: 300)\n"
      << "  --warmup N                     Warmup iterations (default: 50)\n"
      << "  --threads N                    CPU thread count (default: 4)\n";
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help" || key == "-h") {
      Usage();
      std::exit(0);
    }
    if (key.rfind("--", 0) != 0 || i + 1 >= argc) {
      throw std::runtime_error("Invalid argument: " + key);
    }
    const std::string value = argv[++i];
    if (key == "--feather-model") args.feather_model = value;
    else if (key == "--unet-model") args.unet_model = value;
    else if (key == "--wenet-model") args.wenet_model = value;
    else if (key == "--backend") args.backend = value;
    else if (key == "--precision") args.precision = value;
    else if (key == "--iterations") args.iterations = std::stoi(value);
    else if (key == "--warmup") args.warmup = std::stoi(value);
    else if (key == "--threads") args.threads = std::stoi(value);
    else throw std::runtime_error("Unknown option: " + key);
  }
  if (args.feather_model.empty() || args.unet_model.empty()) {
    Usage();
    throw std::runtime_error("Both model paths are required.");
  }
  if (args.backend != "all" && args.backend != "cpu" && args.backend != "metal" &&
      args.backend != "opencl") {
    throw std::runtime_error("--backend must be all, cpu, metal, or opencl.");
  }
  if (args.precision != "high" && args.precision != "normal" && args.precision != "low") {
    throw std::runtime_error("--precision must be high, normal, or low.");
  }
  if (args.iterations <= 0 || args.warmup < 0 || args.threads <= 0) {
    throw std::runtime_error("Invalid numeric option.");
  }
  return args;
}

struct BackendSpec {
  std::string name;
  MNNForwardType forward;
};

MNN::BackendConfig::PrecisionMode ParsePrecision(const std::string& precision) {
  if (precision == "high") return MNN::BackendConfig::Precision_High;
  if (precision == "low") return MNN::BackendConfig::Precision_Low;
  return MNN::BackendConfig::Precision_Normal;
}

std::vector<BackendSpec> RequestedBackends(const Args& args) {
  if (args.backend == "cpu") return {{"CPU", MNN_FORWARD_CPU}};
  if (args.backend == "metal") return {{"Metal", MNN_FORWARD_METAL}};
  if (args.backend == "opencl") return {{"OpenCL", MNN_FORWARD_OPENCL}};
  return {{"CPU", MNN_FORWARD_CPU}, {"Metal", MNN_FORWARD_METAL}, {"OpenCL", MNN_FORWARD_OPENCL}};
}

using TensorPtr = std::unique_ptr<MNN::Tensor, void (*)(MNN::Tensor*)>;

TensorPtr OwnTensor(MNN::Tensor* tensor) {
  return TensorPtr(tensor, MNN::Tensor::destroy);
}

class Model {
 public:
  Model(const std::string& model_path, const BackendSpec& backend, int threads,
        MNN::BackendConfig::PrecisionMode precision, std::vector<std::string> input_names,
        const std::string& output_name)
      : input_names_(std::move(input_names)), output_name_(output_name) {
    interpreter_ = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path.c_str()),
                                                      MNN::Interpreter::destroy);
    if (interpreter_ == nullptr) {
      throw std::runtime_error("Unable to load MNN model: " + model_path);
    }

    // Dynamic waveform inputs must be resized after the session has been created.
    interpreter_->setSessionMode(MNN::Interpreter::Session_Resize_Defer);
    MNN::BackendConfig config_backend;
    config_backend.precision = precision;
    config_backend.power = MNN::BackendConfig::Power_High;
    config_backend.memory = MNN::BackendConfig::Memory_High;
    MNN::ScheduleConfig config;
    config.type = backend.forward;
    config.backupType = MNN_FORWARD_CPU;
    config.numThread = threads;
    config.backendConfig = &config_backend;
    session_ = interpreter_->createSession(config);
    if (session_ == nullptr) {
      throw std::runtime_error("Unable to create MNN session for " + model_path);
    }
    int active_backend = -1;
    if (!interpreter_->getSessionInfo(session_, MNN::Interpreter::BACKENDS, &active_backend)) {
      throw std::runtime_error("Unable to inspect the active MNN backend.");
    }
    const bool cpu_extension = backend.forward == MNN_FORWARD_CPU &&
                               active_backend == MNN_FORWARD_CPU_EXTENSION;
    if (active_backend != backend.forward && !cpu_extension) {
      throw std::runtime_error("MNN requested " + backend.name + " but activated backend " +
                               std::to_string(active_backend));
    }
    for (const auto& name : input_names_) {
      auto* input = interpreter_->getSessionInput(session_, name.c_str());
      if (input == nullptr) {
        throw std::runtime_error("Missing MNN input tensor: " + name);
      }
      inputs_.push_back(input);
    }
    output_ = interpreter_->getSessionOutput(session_, output_name_.c_str());
    if (output_ == nullptr) {
      throw std::runtime_error("Missing MNN output tensor: " + output_name_);
    }
  }

  void BindFloatInput(size_t index, std::vector<int> shape, float* data) {
    if (index >= inputs_.size()) throw std::runtime_error("MNN input index is out of range.");
    interpreter_->resizeTensor(inputs_[index], shape);
    while (host_inputs_.size() <= index) {
      host_inputs_.emplace_back(nullptr, MNN::Tensor::destroy);
    }
    host_inputs_[index] = OwnTensor(MNN::Tensor::create<float>(shape, data, MNN::Tensor::CAFFE));
    if (host_inputs_[index] == nullptr) throw std::runtime_error("Unable to allocate MNN host input.");
  }

  void FinalizeShapes() {
    interpreter_->resizeSession(session_);
  }

  void Run() {
    if (host_inputs_.size() != inputs_.size()) throw std::runtime_error("All MNN inputs must be bound.");
    for (size_t i = 0; i < inputs_.size(); ++i) {
      if (!inputs_[i]->copyFromHostTensor(host_inputs_[i].get())) {
        throw std::runtime_error("Unable to copy MNN input tensor.");
      }
    }
    const auto code = interpreter_->runSession(session_);
    if (code != MNN::NO_ERROR) throw std::runtime_error("MNN inference failed with error code " + std::to_string(code));
    if (host_output_ == nullptr) {
      host_output_ = TensorPtr(new MNN::Tensor(output_, MNN::Tensor::CAFFE), MNN::Tensor::destroy);
    }
    if (!output_->copyToHostTensor(host_output_.get())) {
      throw std::runtime_error("Unable to copy MNN output tensor.");
    }
    volatile float guard = host_output_->host<float>()[0];
    (void)guard;
  }

 private:
  std::shared_ptr<MNN::Interpreter> interpreter_;
  MNN::Session* session_ = nullptr;
  std::vector<std::string> input_names_;
  std::string output_name_;
  std::vector<MNN::Tensor*> inputs_;
  MNN::Tensor* output_ = nullptr;
  std::vector<TensorPtr> host_inputs_;
  TensorPtr host_output_{nullptr, MNN::Tensor::destroy};
};

std::vector<float> MakeSignal(size_t count, float phase) {
  std::vector<float> values(count);
  for (size_t i = 0; i < count; ++i) {
    values[i] = std::sin(static_cast<float>(i) * 0.017F + phase) * 0.5F;
  }
  return values;
}

struct Stats {
  double mean = 0.0;
  double median = 0.0;
  double p95 = 0.0;
};

template <typename Callable>
Stats Measure(const std::string& label, int warmup, int iterations, Callable&& callable) {
  for (int i = 0; i < warmup; ++i) callable();
  std::vector<double> samples;
  samples.reserve(iterations);
  for (int i = 0; i < iterations; ++i) {
    const auto begin = std::chrono::steady_clock::now();
    callable();
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
  }
  std::sort(samples.begin(), samples.end());
  const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  const size_t p95_index = static_cast<size_t>(std::ceil(samples.size() * 0.95)) - 1;
  const Stats result{mean, samples[samples.size() / 2], samples[p95_index]};
  std::cout << std::fixed << std::setprecision(3) << "  " << label << ": mean=" << result.mean
            << " ms, median=" << result.median << " ms, p95=" << result.p95 << " ms\n";
  return result;
}

void BenchmarkBackend(const Args& args, const BackendSpec& backend) {
  const auto precision = ParsePrecision(args.precision);
  Model feather(args.feather_model, backend, args.threads, precision, {"waveform"}, "hidden");
  Model unet(args.unet_model, backend, args.threads, precision, {"input", "audio"}, "output");

  // Two 20 ms HuBERT tokens are one 25 fps video-frame feature.
  auto waveform = MakeSignal(720, 0.0F);
  auto image = MakeSignal(6 * kFaceSize * kFaceSize, 0.1F);
  auto audio = MakeSignal(16 * 32 * 32, 0.2F);
  feather.BindFloatInput(0, {1, 720}, waveform.data());
  feather.FinalizeShapes();
  unet.BindFloatInput(0, {1, 6, kFaceSize, kFaceSize}, image.data());
  unet.BindFloatInput(1, {1, 16, 32, 32}, audio.data());
  unet.FinalizeShapes();

  // Allocate output host tensors and let GPU runtimes finish their first pipeline setup before timing.
  feather.Run();
  unet.Run();

  std::cout << backend.name << " (precision=" << args.precision << ", threads=" << args.threads << ")\n";
  const Stats feather_stats = Measure("FeatherHuBERT [1,720] -> [1,2,1024]", args.warmup,
                                      args.iterations, [&] { feather.Run(); });
  const Stats unet_stats = Measure("UNet one video frame", args.warmup, args.iterations,
                                   [&] { unet.Run(); });
  const double combined = feather_stats.mean + unet_stats.mean;
  std::cout << "  Combined model mean: " << combined << " ms/frame (" << 1000.0 / combined
            << " model-only fps)\n";

  if (!args.wenet_model.empty()) {
    Model wenet(args.wenet_model, backend, args.threads, precision,
                {"chunk", "att_cache", "cnn_cache"}, "output");
    auto chunk = MakeSignal(67 * 80, 0.3F);
    std::vector<float> attention_cache(3 * 8 * 16 * 128, 0.0F);
    std::vector<float> cnn_cache(3 * 1 * 512 * 14, 0.0F);
    wenet.BindFloatInput(0, {1, 1, 67, 80}, chunk.data());
    wenet.BindFloatInput(1, {3, 8, 16, 128}, attention_cache.data());
    wenet.BindFloatInput(2, {3, 1, 512, 14}, cnn_cache.data());
    wenet.FinalizeShapes();
    wenet.Run();
    const Stats wenet_stats = Measure("Wenet encoder [1,1,67,80] -> [1,16,512] (fbank excluded)",
                                      args.warmup, args.iterations, [&] { wenet.Run(); });
    std::cout << "  FeatherHuBERT encoder speedup vs Wenet: " << std::fixed << std::setprecision(2)
              << wenet_stats.mean / feather_stats.mean << "x (MNN model forward only)\n";
  }
}

void Run(const Args& args) {
  std::cout << "MNN " << MNN::getVersion() << " benchmark\n";
  for (const auto& backend : RequestedBackends(args)) {
    try {
      BenchmarkBackend(args, backend);
    } catch (const std::exception& error) {
      std::cout << backend.name << " unavailable: " << error.what() << "\n";
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Run(ParseArgs(argc, argv));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
}
