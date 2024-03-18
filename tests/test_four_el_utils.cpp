//
// Created by Zack Williams on 18/03/2024.
//

#include "../src/four_electron/four_electron_utils.hpp"
#include "catch.hpp"

using test::margin;

TEST_CASE("Test Four Electron Utils - Test energy spin factor") {
  using uw12::four_el::get_energy_spin_factor;

  CHECK_THROWS(get_energy_spin_factor(0, 0, 0, 0, 0));

  CHECK_THROWS(get_energy_spin_factor(1, 1, 0, 1, 0));
  CHECK_THROWS(get_energy_spin_factor(1, 0, 1, 1, 0));
  CHECK_THAT(
      get_energy_spin_factor(1, 0, 0, 1, 0),
      Catch::Matchers::WithinAbs(2, margin)
  );
  CHECK_THAT(
      get_energy_spin_factor(1, 0, 0, 0, 1),
      Catch::Matchers::WithinAbs(2, margin)
  );
  CHECK_THAT(
      get_energy_spin_factor(1, 0, 0, 1, 0.5),
      Catch::Matchers::WithinAbs(3, margin)
  );

  CHECK_THROWS(get_energy_spin_factor(2, 2, 0, 1, 0));
  CHECK_THROWS(get_energy_spin_factor(2, 0, 2, 1, 0));
  CHECK_THAT(
      get_energy_spin_factor(2, 0, 0, 1, 0.5),
      Catch::Matchers::WithinAbs(0.5, margin)
  );
  CHECK_THAT(
      get_energy_spin_factor(2, 0, 1, 1, 0.5),
      Catch::Matchers::WithinAbs(1, margin)
  );
  CHECK_THAT(
      get_energy_spin_factor(2, 1, 0, 1, 0.5),
      Catch::Matchers::WithinAbs(1, margin)
  );
  CHECK_THAT(
      get_energy_spin_factor(2, 1, 1, 1, 0.5),
      Catch::Matchers::WithinAbs(0.5, margin)
  );

  CHECK_THROWS(get_energy_spin_factor(3, 0, 0, 0, 0));
}
