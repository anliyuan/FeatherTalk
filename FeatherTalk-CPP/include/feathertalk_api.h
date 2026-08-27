#pragma once

#include <string>

namespace feathertalk {

struct OfflineOptions {
  std::string feather_model;
  std::string visual_model;
  std::string dataset;
  std::string audio_wav;
  std::string output_mp4;
  std::string frames_dir;

  std::string backend = "cpu";
  std::string precision = "low";
  int threads = 1;
  int max_frames = 0;
  int video_crf = 18;
  bool profile = false;
};

bool RunOffline(const OfflineOptions& options, std::string* error = nullptr);
int RunCommandLine(int argc, char** argv);

}  // namespace feathertalk
