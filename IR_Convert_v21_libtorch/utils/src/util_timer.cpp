#include <util_timer.h>

namespace core
{
  Timer::Timer(std::string name) : name(name) {}

  void Timer::start()
  {
    start_time = std::chrono::high_resolution_clock::now();
  }

  void Timer::stop()
  {
    const auto elapsed = std::chrono::high_resolution_clock::now() - start_time;
    const auto period = std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count();
    total_time += period;
    count++;
  }

  void Timer::show()
  {
    std::cout << "[" << name << " Time]" << std::endl;
    std::cout << "\t   All: " << total_time << std::endl;
    std::cout << "\tSingle: " << total_time * 1000 / count << std::endl;
    std::cout << "\t   FPS: " << count / total_time << std::endl;
    std::cout << "\t Count: " << count << std::endl;
  }
} /* namespace core */
