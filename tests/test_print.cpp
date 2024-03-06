//
// Created by Zack Williams on 13/02/2024.
//

#include "../src/utils/print.hpp"
#include "catch.hpp"

using namespace uw12::print;

#include <iostream>
#include <sstream>

/// \brief Class to deal with restoring iostream output following redirection to
/// buffer Solution from
/// https://stackoverflow.com/questions/74755467/unable-to-use-redirected-stdcout-in-catch2
class AutoRestoreRdbuf {
  std::ostream &out;
  std::streambuf *old;

 public:
  /// \brief Constructor which sets output to out while saving old RdBuf to be
  /// restored at destruction \param out output stream
  explicit AutoRestoreRdbuf(std::ostream &out) : out{out}, old{out.rdbuf()} {}

  /// \brief Const copy constructor
  AutoRestoreRdbuf(const AutoRestoreRdbuf &) = delete;

  /// \brief Non-const copy constructor
  AutoRestoreRdbuf(AutoRestoreRdbuf &&) = delete;

  /// \brief Destructor which restores old output path
  ~AutoRestoreRdbuf() { out.rdbuf(old); }
};

/// \brief Function which redirects output to osstringsteam and captures it
/// while function is called
///
/// \param function The function from which output will be captured
/// \param out The output method (std::cout)
///
/// \return The function output as a string
std::string write_output_to_string(
    const std::function<void()> &function, std::ostream &out = std::cout
) {
  AutoRestoreRdbuf restore(out);
  const std::ostringstream oss;
  std::cout.rdbuf(oss.rdbuf());
  function();
  return oss.str();
}

TEST_CASE("Test print - print_character_line") {
  const auto output =
      write_output_to_string([]() { print_character_line('x', 10); });

  CHECK(output == "xxxxxxxxxx\n");  // TODO Check portability
}

TEST_CASE("Test Print - print_header") {
  const auto header =
      write_output_to_string([]() { print_header("This is a header"); });

  std::string expected;
  expected = expected + "\n================================================\n" +
             " This is a header\n" +
             "================================================\n";

  CHECK(header == expected);
}

TEST_CASE("Test Print - print_result") {
  CHECK_THROWS_WITH(
      print_result(
          "This is my result, my result is very long it needs explaining",
          0.1234567890
      ),
      "Output string exceeds size of print block"
  );

  CHECK_NOTHROW(print_result(
      "This is my result, my result is very long it needs explaining",
      0.1234567890,
      64
  ));

  const auto result = write_output_to_string([]() {
    print_result("This is my result", 1.234567890);
  });

  const std::string expected =
      " This is my result:                   1.234568\n";

  CHECK(result == expected);
}
