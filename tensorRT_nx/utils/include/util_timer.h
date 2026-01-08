#ifndef INCLUDE_CORE_TIMER_H_
#define INCLUDE_CORE_TIMER_H_

#include <chrono>
#include <iostream>
#include <string>

namespace core {

class Timer {
 public:
  explicit Timer(const std::string& name);

  void start();
  void stop();
  void show();

 private:
  std::string name_;
  int count_ = 0;
  double total_time_ = 0;
  std::chrono::time_point<std::chrono::system_clock> start_time_;
};

}  // namespace core

#endif  // INCLUDE_CORE_TIMER_H_
