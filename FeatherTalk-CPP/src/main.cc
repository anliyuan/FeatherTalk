#include <MNN/ErrorCode.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <sstream>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kSampleRate = 16000;
constexpr int kFeatureDim = 1024;
constexpr int kTokensPerFrame = 2;
constexpr int kAudioWindowFrames = 8;
constexpr int kAudioHalfWindow = 4;
constexpr int kFaceCropSize = 168;
constexpr int kFaceInnerSize = 160;
constexpr int kFaceBorder = 4;
constexpr int kOutputFps = 25;

struct Args {
  fs::path feather_model;
  fs::path unet_model;
  fs::path dataset;
  fs::path audio;
  fs::path output;
  fs::path frames_dir;
  int max_frames = 0;
  int threads = 4;
  std::string backend = "metal";
  std::string precision = "high";
};

struct Image {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> bgr;

  uint8_t* Pixel(int x, int y) { return bgr.data() + (y * width + x) * 3; }
  const uint8_t* Pixel(int x, int y) const { return bgr.data() + (y * width + x) * 3; }
};

struct Bbox {
  int x = 0;
  int y = 0;
  int width = 0;
  int height = 0;
};

struct FrameAsset {
  fs::path image;
  fs::path landmarks;
  int index = 0;
};

struct WavData {
  int sample_rate = 0;
  std::vector<float> samples;
};

std::string ToLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

bool IsFlag(const std::string& value) { return value.rfind("--", 0) == 0; }

void PrintUsage() {
  std::cout
      << "FeatherTalk C++ offline inference\n\n"
      << "Required:\n"
      << "  --feather-model PATH   FeatherHuBERT model ([1, samples] -> [1, T, 1024])\n"
      << "  --unet-model PATH      FeatherTalk UNet model\n"
      << "  --dataset PATH         Directory with full_body_img/ and landmarks/\n"
      << "  --audio PATH           16 kHz PCM/float WAV\n"
      << "  --output PATH          Output MP4 path\n\n"
      << "Optional:\n"
      << "  --frames-dir PATH      Also save rendered PNG frames in this directory\n"
      << "  --max-frames N         Render only the first N frames (0 means all)\n"
      << "  --threads N            CPU thread count (default: 4)\n"
      << "  --backend cpu|metal|opencl  MNN backend (default: metal)\n"
      << "  --precision high|normal|low  MNN precision (default: high)\n";
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help" || key == "-h") {
      PrintUsage();
      std::exit(0);
    }
    if (!IsFlag(key) || i + 1 >= argc) {
      throw std::runtime_error("Invalid argument: " + key);
    }
    const std::string value = argv[++i];
    if (key == "--feather-model") {
      args.feather_model = value;
    } else if (key == "--unet-model") {
      args.unet_model = value;
    } else if (key == "--dataset") {
      args.dataset = value;
    } else if (key == "--audio") {
      args.audio = value;
    } else if (key == "--output") {
      args.output = value;
    } else if (key == "--frames-dir") {
      args.frames_dir = value;
    } else if (key == "--max-frames") {
      args.max_frames = std::stoi(value);
    } else if (key == "--threads") {
      args.threads = std::stoi(value);
    } else if (key == "--backend") {
      args.backend = value;
    } else if (key == "--precision") {
      args.precision = value;
    } else {
      throw std::runtime_error("Unknown option: " + key);
    }
  }

  if (args.feather_model.empty() || args.unet_model.empty() || args.dataset.empty() ||
      args.audio.empty() || args.output.empty()) {
    PrintUsage();
    throw std::runtime_error("All required arguments must be provided.");
  }
  if (args.max_frames < 0 || args.threads <= 0) {
    throw std::runtime_error("--max-frames must be >= 0 and --threads must be > 0.");
  }
  if (args.backend != "cpu" && args.backend != "metal" && args.backend != "opencl") {
    throw std::runtime_error("--backend must be cpu, metal, or opencl.");
  }
  if (args.precision != "high" && args.precision != "normal" && args.precision != "low") {
    throw std::runtime_error("--precision must be high, normal, or low.");
  }
  return args;
}

void EnsureFile(const fs::path& path, const std::string& label) {
  if (!fs::is_regular_file(path)) {
    throw std::runtime_error(label + " not found: " + path.string());
  }
}

uint16_t ReadU16(const uint8_t* data) {
  return static_cast<uint16_t>(data[0]) | (static_cast<uint16_t>(data[1]) << 8);
}

uint32_t ReadU32(const uint8_t* data) {
  return static_cast<uint32_t>(data[0]) | (static_cast<uint32_t>(data[1]) << 8) |
         (static_cast<uint32_t>(data[2]) << 16) | (static_cast<uint32_t>(data[3]) << 24);
}

WavData ReadWavMono(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Cannot open WAV: " + path.string());
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)), {});
  if (bytes.size() < 12 || std::memcmp(bytes.data(), "RIFF", 4) != 0 ||
      std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
    throw std::runtime_error("Only RIFF/WAVE input is supported: " + path.string());
  }

  uint16_t format = 0;
  uint16_t channels = 0;
  uint16_t bits_per_sample = 0;
  uint32_t sample_rate = 0;
  uint16_t block_align = 0;
  const uint8_t* pcm = nullptr;
  size_t pcm_bytes = 0;

  size_t offset = 12;
  while (offset + 8 <= bytes.size()) {
    const uint8_t* chunk = bytes.data() + offset;
    const uint32_t chunk_size = ReadU32(chunk + 4);
    const size_t payload = offset + 8;
    const size_t next = payload + chunk_size + (chunk_size & 1U);
    if (next > bytes.size()) throw std::runtime_error("Malformed WAV chunk in " + path.string());
    if (std::memcmp(chunk, "fmt ", 4) == 0) {
      if (chunk_size < 16) throw std::runtime_error("Invalid WAV fmt chunk.");
      format = ReadU16(bytes.data() + payload);
      channels = ReadU16(bytes.data() + payload + 2);
      sample_rate = ReadU32(bytes.data() + payload + 4);
      block_align = ReadU16(bytes.data() + payload + 12);
      bits_per_sample = ReadU16(bytes.data() + payload + 14);
    } else if (std::memcmp(chunk, "data", 4) == 0) {
      pcm = bytes.data() + payload;
      pcm_bytes = chunk_size;
    }
    offset = next;
  }

  if (pcm == nullptr || channels == 0 || block_align == 0) {
    throw std::runtime_error("WAV must contain fmt and data chunks: " + path.string());
  }
  if (sample_rate != kSampleRate) {
    throw std::runtime_error("Expected a 16 kHz WAV, got " + std::to_string(sample_rate) +
                             " Hz. Convert with ffmpeg -ar 16000 -ac 1.");
  }

  const size_t frames = pcm_bytes / block_align;
  WavData result;
  result.sample_rate = static_cast<int>(sample_rate);
  result.samples.reserve(frames);
  const size_t sample_bytes = bits_per_sample / 8;
  for (size_t frame = 0; frame < frames; ++frame) {
    const uint8_t* sample = pcm + frame * block_align;
    float value = 0.0F;
    if (format == 1 && bits_per_sample == 16) {
      const int16_t raw = static_cast<int16_t>(ReadU16(sample));
      value = static_cast<float>(raw) / 32768.0F;
    } else if (format == 1 && bits_per_sample == 24) {
      int32_t raw = static_cast<int32_t>(sample[0]) | (static_cast<int32_t>(sample[1]) << 8) |
                    (static_cast<int32_t>(sample[2]) << 16);
      if (raw & 0x00800000) raw |= 0xFF000000;
      value = static_cast<float>(raw) / 8388608.0F;
    } else if (format == 1 && bits_per_sample == 32) {
      int32_t raw = 0;
      std::memcpy(&raw, sample, sizeof(raw));
      value = static_cast<float>(raw) / 2147483648.0F;
    } else if (format == 3 && bits_per_sample == 32) {
      std::memcpy(&value, sample, sizeof(value));
    } else {
      throw std::runtime_error("Unsupported WAV format=" + std::to_string(format) +
                               ", bits=" + std::to_string(bits_per_sample) +
                               ", sample_bytes=" + std::to_string(sample_bytes));
    }
    result.samples.push_back(value);  // Match Python: retain channel 0 for stereo input.
  }
  return result;
}

void NormalizeWaveform(std::vector<float>* waveform) {
  if (waveform->empty()) throw std::runtime_error("Input waveform is empty.");
  const double mean = std::accumulate(waveform->begin(), waveform->end(), 0.0) /
                      static_cast<double>(waveform->size());
  double variance = 0.0;
  for (const float sample : *waveform) {
    const double delta = static_cast<double>(sample) - mean;
    variance += delta * delta;
  }
  variance /= static_cast<double>(waveform->size());
  const float scale = static_cast<float>(1.0 / std::sqrt(variance + 1e-7));
  for (float& sample : *waveform) sample = (sample - static_cast<float>(mean)) * scale;
}

Image LoadImageBgr(const fs::path& path) {
  int width = 0;
  int height = 0;
  int channels = 0;
  stbi_uc* rgb = stbi_load(path.string().c_str(), &width, &height, &channels, 3);
  if (rgb == nullptr) {
    throw std::runtime_error("Cannot read image " + path.string() + ": " + stbi_failure_reason());
  }
  Image image;
  image.width = width;
  image.height = height;
  image.bgr.resize(static_cast<size_t>(width) * height * 3);
  for (int i = 0; i < width * height; ++i) {
    image.bgr[i * 3] = rgb[i * 3 + 2];
    image.bgr[i * 3 + 1] = rgb[i * 3 + 1];
    image.bgr[i * 3 + 2] = rgb[i * 3];
  }
  stbi_image_free(rgb);
  return image;
}

Image CropImage(const Image& image, const Bbox& bbox) {
  if (bbox.width <= 0 || bbox.height <= 0 || bbox.x < 0 || bbox.y < 0 ||
      bbox.x + bbox.width > image.width || bbox.y + bbox.height > image.height) {
    throw std::runtime_error("Face bbox is outside image bounds.");
  }
  Image crop;
  crop.width = bbox.width;
  crop.height = bbox.height;
  crop.bgr.resize(static_cast<size_t>(crop.width) * crop.height * 3);
  for (int y = 0; y < crop.height; ++y) {
    std::memcpy(crop.Pixel(0, y), image.Pixel(bbox.x, bbox.y + y),
                static_cast<size_t>(crop.width) * 3);
  }
  return crop;
}

Image ResizeBilinear(const Image& source, int target_width, int target_height) {
  if (target_width <= 0 || target_height <= 0) throw std::runtime_error("Invalid resize target.");
  Image result;
  result.width = target_width;
  result.height = target_height;
  result.bgr.resize(static_cast<size_t>(target_width) * target_height * 3);
  const float scale_x = static_cast<float>(source.width) / target_width;
  const float scale_y = static_cast<float>(source.height) / target_height;
  for (int y = 0; y < target_height; ++y) {
    const float source_y = (y + 0.5F) * scale_y - 0.5F;
    const int y0 = std::clamp(static_cast<int>(std::floor(source_y)), 0, source.height - 1);
    const int y1 = std::clamp(y0 + 1, 0, source.height - 1);
    const float wy = std::clamp(source_y - std::floor(source_y), 0.0F, 1.0F);
    for (int x = 0; x < target_width; ++x) {
      const float source_x = (x + 0.5F) * scale_x - 0.5F;
      const int x0 = std::clamp(static_cast<int>(std::floor(source_x)), 0, source.width - 1);
      const int x1 = std::clamp(x0 + 1, 0, source.width - 1);
      const float wx = std::clamp(source_x - std::floor(source_x), 0.0F, 1.0F);
      uint8_t* dst = result.Pixel(x, y);
      const uint8_t* p00 = source.Pixel(x0, y0);
      const uint8_t* p01 = source.Pixel(x1, y0);
      const uint8_t* p10 = source.Pixel(x0, y1);
      const uint8_t* p11 = source.Pixel(x1, y1);
      for (int c = 0; c < 3; ++c) {
        const float top = p00[c] * (1.0F - wx) + p01[c] * wx;
        const float bottom = p10[c] * (1.0F - wx) + p11[c] * wx;
        dst[c] = static_cast<uint8_t>(std::lround(top * (1.0F - wy) + bottom * wy));
      }
    }
  }
  return result;
}

void PasteImage(Image* destination, const Image& source, int x, int y) {
  if (x < 0 || y < 0 || x + source.width > destination->width ||
      y + source.height > destination->height) {
    throw std::runtime_error("Paste is outside image bounds.");
  }
  for (int row = 0; row < source.height; ++row) {
    std::memcpy(destination->Pixel(x, y + row), source.Pixel(0, row),
                static_cast<size_t>(source.width) * 3);
  }
}

std::vector<std::array<int, 2>> ReadLandmarks(const fs::path& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("Cannot read landmarks: " + path.string());
  std::vector<std::array<int, 2>> points;
  float x = 0.0F;
  float y = 0.0F;
  while (input >> x >> y) points.push_back({static_cast<int>(x), static_cast<int>(y)});
  if (points.size() <= 52) throw std::runtime_error("Need at least 53 landmarks: " + path.string());
  return points;
}

Bbox ComputeFaceBbox(const fs::path& landmarks) {
  const auto points = ReadLandmarks(landmarks);
  const int xmin = points[1][0];
  const int ymin = points[52][1];
  const int xmax = points[31][0];
  const int width = xmax - xmin;
  return {xmin, ymin, width, width};
}

std::optional<int> NumericStem(const fs::path& path) {
  const std::string stem = path.stem().string();
  if (stem.empty() || !std::all_of(stem.begin(), stem.end(), ::isdigit)) return std::nullopt;
  try {
    return std::stoi(stem);
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

std::vector<FrameAsset> LoadAssets(const fs::path& dataset) {
  const fs::path image_dir = dataset / "full_body_img";
  const fs::path landmark_dir = dataset / "landmarks";
  if (!fs::is_directory(image_dir) || !fs::is_directory(landmark_dir)) {
    throw std::runtime_error("Dataset must contain full_body_img/ and landmarks/: " + dataset.string());
  }
  std::map<int, fs::path> images;
  for (const auto& entry : fs::directory_iterator(image_dir)) {
    if (!entry.is_regular_file()) continue;
    const std::string extension = ToLower(entry.path().extension().string());
    if (extension != ".jpg" && extension != ".jpeg" && extension != ".png") continue;
    if (const auto index = NumericStem(entry.path())) images[*index] = entry.path();
  }
  std::map<int, fs::path> landmarks;
  for (const auto& entry : fs::directory_iterator(landmark_dir)) {
    if (!entry.is_regular_file() || ToLower(entry.path().extension().string()) != ".lms") continue;
    if (const auto index = NumericStem(entry.path())) landmarks[*index] = entry.path();
  }
  std::vector<FrameAsset> assets;
  for (const auto& [index, image] : images) {
    const auto landmark = landmarks.find(index);
    if (landmark != landmarks.end()) assets.push_back({image, landmark->second, index});
  }
  if (assets.size() < 2) throw std::runtime_error("Need at least two image/landmark frame pairs.");
  return assets;
}

class BouncePicker {
 public:
  explicit BouncePicker(size_t count) : count_(count) {}

  size_t Next() {
    if (index_ >= static_cast<int>(count_) - 1) step_ = -1;
    if (index_ <= 0) step_ = 1;
    index_ += step_;
    return static_cast<size_t>(index_);
  }

 private:
  size_t count_ = 0;
  int index_ = 0;
  int step_ = 0;
};

MNNForwardType ParseMnnBackend(const std::string& backend) {
  if (backend == "cpu") return MNN_FORWARD_CPU;
  if (backend == "metal") return MNN_FORWARD_METAL;
  return MNN_FORWARD_OPENCL;
}

MNN::BackendConfig::PrecisionMode ParseMnnPrecision(const std::string& precision) {
  if (precision == "high") return MNN::BackendConfig::Precision_High;
  if (precision == "low") return MNN::BackendConfig::Precision_Low;
  return MNN::BackendConfig::Precision_Normal;
}

class MnnModel {
 public:
  MnnModel(const fs::path& model_path,
           const std::vector<std::string>& input_names,
           const std::string& output_name,
           const Args& args)
      : input_names_(input_names), output_name_(output_name) {
    interpreter_ = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path.c_str()),
                                                      MNN::Interpreter::destroy);
    if (interpreter_ == nullptr) throw std::runtime_error("Unable to load MNN model: " + model_path.string());

    interpreter_->setSessionMode(MNN::Interpreter::Session_Resize_Defer);
    MNN::BackendConfig backend_config;
    backend_config.precision = ParseMnnPrecision(args.precision);
    backend_config.power = MNN::BackendConfig::Power_High;
    backend_config.memory = MNN::BackendConfig::Memory_High;
    MNN::ScheduleConfig config;
    config.type = ParseMnnBackend(args.backend);
    config.backupType = MNN_FORWARD_CPU;
    config.numThread = args.threads;
    config.backendConfig = &backend_config;
    session_ = interpreter_->createSession(config);
    if (session_ == nullptr) throw std::runtime_error("Unable to create MNN session.");

    int active_backend = -1;
    if (!interpreter_->getSessionInfo(session_, MNN::Interpreter::BACKENDS, &active_backend)) {
      throw std::runtime_error("Unable to inspect the active MNN backend.");
    }
    const bool cpu_extension = config.type == MNN_FORWARD_CPU &&
                               active_backend == MNN_FORWARD_CPU_EXTENSION;
    if (active_backend != config.type && !cpu_extension) {
      throw std::runtime_error("MNN requested " + args.backend + " but activated backend " +
                               std::to_string(active_backend));
    }
    for (const auto& name : input_names_) {
      MNN::Tensor* input = interpreter_->getSessionInput(session_, name.c_str());
      if (input == nullptr) throw std::runtime_error("Missing MNN input: " + name);
      inputs_.push_back(input);
    }
    output_ = interpreter_->getSessionOutput(session_, output_name_.c_str());
    if (output_ == nullptr) throw std::runtime_error("Missing MNN output: " + output_name_);
  }

  std::vector<float> RunSingle(const std::vector<float>& values,
                               const std::vector<int64_t>& shape,
                               std::vector<int64_t>* output_shape) {
    return Run({&values}, {ToMnnShape(shape)}, output_shape);
  }

  std::vector<float> RunUnet(const std::vector<float>& image,
                             const std::vector<float>& audio,
                             std::vector<int64_t>* output_shape) {
    return Run({&image, &audio}, {{1, 6, kFaceInnerSize, kFaceInnerSize}, {1, 16, 32, 32}}, output_shape);
  }

 private:
  using TensorPtr = std::unique_ptr<MNN::Tensor, void (*)(MNN::Tensor*)>;

  static std::vector<int> ToMnnShape(const std::vector<int64_t>& shape) {
    std::vector<int> result;
    result.reserve(shape.size());
    for (const int64_t value : shape) result.push_back(static_cast<int>(value));
    return result;
  }

  std::vector<float> Run(const std::vector<const std::vector<float>*>& values,
                         const std::vector<std::vector<int>>& shapes,
                         std::vector<int64_t>* output_shape) {
    if (values.size() != inputs_.size() || shapes.size() != inputs_.size()) {
      throw std::runtime_error("MNN input count does not match the model.");
    }
    if (shapes != input_shapes_) {
      for (size_t i = 0; i < inputs_.size(); ++i) interpreter_->resizeTensor(inputs_[i], shapes[i]);
      interpreter_->resizeSession(session_);
      input_shapes_ = shapes;
    }

    std::vector<TensorPtr> host_inputs;
    host_inputs.reserve(inputs_.size());
    for (size_t i = 0; i < inputs_.size(); ++i) {
      auto host = TensorPtr(MNN::Tensor::create<float>(shapes[i],
                                                        const_cast<float*>(values[i]->data()),
                                                        MNN::Tensor::CAFFE),
                            MNN::Tensor::destroy);
      if (host == nullptr || !inputs_[i]->copyFromHostTensor(host.get())) {
        throw std::runtime_error("Unable to copy MNN input tensor.");
      }
      host_inputs.push_back(std::move(host));
    }
    const MNN::ErrorCode code = interpreter_->runSession(session_);
    if (code != MNN::NO_ERROR) throw std::runtime_error("MNN inference failed with error code " + std::to_string(code));

    MNN::Tensor output_host(output_, MNN::Tensor::CAFFE);
    if (!output_->copyToHostTensor(&output_host)) {
      throw std::runtime_error("Unable to copy MNN output tensor.");
    }
    const std::vector<int> shape = output_->shape();
    output_shape->assign(shape.begin(), shape.end());
    const float* data = output_host.host<float>();
    return {data, data + output_host.elementSize()};
  }

  std::shared_ptr<MNN::Interpreter> interpreter_;
  MNN::Session* session_ = nullptr;
  std::vector<std::string> input_names_;
  std::string output_name_;
  std::vector<MNN::Tensor*> inputs_;
  MNN::Tensor* output_ = nullptr;
  std::vector<std::vector<int>> input_shapes_;
};

std::vector<float> ExtractFeatherFeatures(MnnModel& encoder, std::vector<float> waveform) {
  NormalizeWaveform(&waveform);
  std::vector<int64_t> output_shape;
  const std::vector<float> output = encoder.RunSingle(
      waveform, {1, static_cast<int64_t>(waveform.size())}, &output_shape);
  if (output_shape.size() != 3 || output_shape[0] != 1 || output_shape[2] != kFeatureDim) {
    throw std::runtime_error("Expected FeatherHuBERT output [1, T, 1024].");
  }
  int64_t token_count = output_shape[1];
  if (token_count < 2) throw std::runtime_error("Audio is too short for FeatherHuBERT.");
  if (token_count % 2 != 0) --token_count;  // Match feather_hubert.make_even_first_dim.
  return {output.begin(), output.begin() + token_count * kFeatureDim};
}

std::vector<float> GatherAudioWindow(const std::vector<float>& features, int video_frame) {
  const int total_frames = static_cast<int>(features.size() / (kTokensPerFrame * kFeatureDim));
  std::vector<float> window(kAudioWindowFrames * kTokensPerFrame * kFeatureDim, 0.0F);
  for (int window_frame = 0; window_frame < kAudioWindowFrames; ++window_frame) {
    const int source_frame = video_frame - kAudioHalfWindow + window_frame;
    if (source_frame < 0 || source_frame >= total_frames) continue;
    const size_t source = static_cast<size_t>(source_frame) * kTokensPerFrame * kFeatureDim;
    const size_t destination = static_cast<size_t>(window_frame) * kTokensPerFrame * kFeatureDim;
    std::copy_n(features.begin() + source, kTokensPerFrame * kFeatureDim, window.begin() + destination);
  }
  return window;
}

std::vector<float> BuildImageInput(const Image& face_crop) {
  if (face_crop.width != kFaceCropSize || face_crop.height != kFaceCropSize) {
    throw std::runtime_error("Face crop must be 168x168.");
  }
  constexpr int kPlane = kFaceInnerSize * kFaceInnerSize;
  std::vector<float> input(6 * kPlane, 0.0F);
  for (int y = 0; y < kFaceInnerSize; ++y) {
    for (int x = 0; x < kFaceInnerSize; ++x) {
      const uint8_t* pixel = face_crop.Pixel(x + kFaceBorder, y + kFaceBorder);
      const bool masked = x >= 5 && x < 155 && y >= 5 && y < 150;
      const int offset = y * kFaceInnerSize + x;
      for (int c = 0; c < 3; ++c) {
        input[c * kPlane + offset] = pixel[c] / 255.0F;
        input[(c + 3) * kPlane + offset] = masked ? 0.0F : pixel[c] / 255.0F;
      }
    }
  }
  return input;
}

uint8_t ToByte(float value) {
  const float scaled = std::clamp(value * 255.0F, 0.0F, 255.0F);
  return static_cast<uint8_t>(std::lround(scaled));
}

Image RenderFrame(MnnModel& unet,
                  Image image,
                  const Bbox& bbox,
                  const std::vector<float>& audio_window) {
  Image face_crop = ResizeBilinear(CropImage(image, bbox), kFaceCropSize, kFaceCropSize);
  const std::vector<float> image_input = BuildImageInput(face_crop);
  std::vector<int64_t> output_shape;
  const std::vector<float> prediction = unet.RunUnet(image_input, audio_window, &output_shape);
  if (output_shape != std::vector<int64_t>({1, 3, kFaceInnerSize, kFaceInnerSize})) {
    throw std::runtime_error("Expected UNet output [1, 3, 160, 160].");
  }
  constexpr int kPlane = kFaceInnerSize * kFaceInnerSize;
  for (int y = 0; y < kFaceInnerSize; ++y) {
    for (int x = 0; x < kFaceInnerSize; ++x) {
      uint8_t* pixel = face_crop.Pixel(x + kFaceBorder, y + kFaceBorder);
      const int offset = y * kFaceInnerSize + x;
      pixel[0] = ToByte(prediction[offset]);
      pixel[1] = ToByte(prediction[kPlane + offset]);
      pixel[2] = ToByte(prediction[2 * kPlane + offset]);
    }
  }
  PasteImage(&image, ResizeBilinear(face_crop, bbox.width, bbox.height), bbox.x, bbox.y);
  return image;
}

std::string ShellQuote(const fs::path& path) {
  std::string value = path.string();
  std::string result = "'";
  for (const char ch : value) {
    if (ch == '\'') result += "'\\\"'\\\"'";
    else result += ch;
  }
  result += "'";
  return result;
}

FILE* StartFfmpeg(const Args& args, int width, int height) {
  if (!args.output.parent_path().empty()) fs::create_directories(args.output.parent_path());
  const std::string command =
      "ffmpeg -y -loglevel error -f rawvideo -pix_fmt bgr24 -video_size " +
      std::to_string(width) + "x" + std::to_string(height) + " -framerate " +
      std::to_string(kOutputFps) + " -i - -i " + ShellQuote(args.audio) +
      " -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest " + ShellQuote(args.output);
  std::cout << "[ffmpeg] encoding " << args.output << '\n';
  FILE* pipe = popen(command.c_str(), "w");
  if (pipe == nullptr) throw std::runtime_error("Cannot start ffmpeg. Ensure it is on PATH.");
  return pipe;
}

void WriteFrame(FILE* pipe, const Image& image) {
  const size_t expected = image.bgr.size();
  if (std::fwrite(image.bgr.data(), 1, expected, pipe) != expected) {
    throw std::runtime_error("Failed to write a frame to ffmpeg.");
  }
}

void WritePngFrame(const fs::path& directory, int frame_index, const Image& image) {
  fs::create_directories(directory);
  std::ostringstream name;
  name << std::setfill('0') << std::setw(6) << frame_index << ".png";
  const fs::path output = directory / name.str();

  std::vector<uint8_t> rgb(image.bgr.size());
  for (size_t pixel = 0; pixel < image.bgr.size() / 3; ++pixel) {
    rgb[pixel * 3] = image.bgr[pixel * 3 + 2];
    rgb[pixel * 3 + 1] = image.bgr[pixel * 3 + 1];
    rgb[pixel * 3 + 2] = image.bgr[pixel * 3];
  }
  if (stbi_write_png(output.string().c_str(), image.width, image.height, 3, rgb.data(), image.width * 3) == 0) {
    throw std::runtime_error("Failed to write PNG frame: " + output.string());
  }
}

void CloseFfmpeg(FILE* pipe) {
  const int status = pclose(pipe);
  if (status != 0) throw std::runtime_error("ffmpeg exited with status " + std::to_string(status));
}

void RunPipeline(const Args& args,
                 const std::vector<FrameAsset>& assets,
                 const WavData& wav,
                 MnnModel& feather,
                 MnnModel& unet) {
  std::vector<float> features = ExtractFeatherFeatures(feather, wav.samples);
  const int total_frames = static_cast<int>(features.size() / (kTokensPerFrame * kFeatureDim));
  int render_frames = total_frames;
  if (args.max_frames > 0) render_frames = std::min(render_frames, args.max_frames);
  std::cout << "[features] " << total_frames << " video frames; rendering " << render_frames << "\n";

  const Image first_image = LoadImageBgr(assets.front().image);
  if (!args.frames_dir.empty()) {
    std::cout << "[frames] saving PNG frames to " << args.frames_dir << '\n';
  }
  FILE* ffmpeg = StartFfmpeg(args, first_image.width, first_image.height);
  try {
    BouncePicker picker(assets.size());
    for (int frame = 0; frame < render_frames; ++frame) {
      const FrameAsset& asset = assets[picker.Next()];
      Image image = LoadImageBgr(asset.image);
      if (image.width != first_image.width || image.height != first_image.height) {
        throw std::runtime_error("All dataset images must share the same resolution.");
      }
      const Image rendered = RenderFrame(unet, std::move(image), ComputeFaceBbox(asset.landmarks),
                                         GatherAudioWindow(features, frame));
      if (!args.frames_dir.empty()) WritePngFrame(args.frames_dir, frame, rendered);
      WriteFrame(ffmpeg, rendered);
      if ((frame + 1) % 25 == 0 || frame + 1 == render_frames) {
        std::cout << "\r[render] " << (frame + 1) << "/" << render_frames << std::flush;
      }
    }
    std::cout << '\n';
    CloseFfmpeg(ffmpeg);
  } catch (...) {
    pclose(ffmpeg);
    throw;
  }
  std::cout << "[done] " << args.output << '\n';
}

void Run(const Args& args) {
  EnsureFile(args.feather_model, "FeatherHuBERT model");
  EnsureFile(args.unet_model, "UNet model");
  EnsureFile(args.audio, "Audio WAV");

  const std::vector<FrameAsset> assets = LoadAssets(args.dataset);
  const WavData wav = ReadWavMono(args.audio);
  std::cout << "[audio] " << wav.samples.size() << " samples, "
            << static_cast<double>(wav.samples.size()) / wav.sample_rate << " seconds\n";

  std::cout << "[mnn] backend=" << args.backend << ", precision=" << args.precision
            << ", threads=" << args.threads << '\n';
  MnnModel feather(args.feather_model, {"waveform"}, "hidden", args);
  MnnModel unet(args.unet_model, {"input", "audio"}, "output", args);
  RunPipeline(args, assets, wav, feather, unet);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Run(ParseArgs(argc, argv));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
  }
  return 1;
}
