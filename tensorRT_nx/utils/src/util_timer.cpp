#include <util_timer.h>

namespace core {

Timer::Timer(const std::string& name) : name_(name)
{
}

void Timer::start()
{
  start_time_ = std::chrono::high_resolution_clock::now();
}

void Timer::stop()
{
  const auto elapsed = std::chrono::high_resolution_clock::now() - start_time_;
  const auto period =
      std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count();
  total_time_ += period;
  count_++;
}

void Timer::show()
{
  std::cout << "[" << name_ << " Time]" << std::endl;
  std::cout << "\t   All: " << total_time_ << std::endl;
  std::cout << "\tSingle: " << total_time_ * 1000 / count_ << std::endl;
  std::cout << "\t   FPS: " << count_ / total_time_ << std::endl;
  std::cout << "\t Count: " << count_ << std::endl;
}

}  // namespace core
