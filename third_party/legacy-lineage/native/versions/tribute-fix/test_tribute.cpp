#include "ygoenv/ygopro/ygopro.h"

#include <cstdlib>
#include <iostream>

using namespace ygopro;

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

class TributeFixture : public YGOProEnvImpl {
 public:
  TributeFixture() : YGOProEnvImpl(YGOProEnvSpec(), 37) {
    pduel_ = YGO_CreateDuel(37);
    std::fill(std::begin(resp_buf_), std::end(resp_buf_), 0x7f);
  }

  ~TributeFixture() {
    YGO_EndDuel(pduel_);
  }

  void message(int minimum, int maximum, const std::vector<int>& weights,
               int player = 0) {
    std::vector<byte> bytes = {
        MSG_SELECT_TRIBUTE, static_cast<byte>(player), 0,
        static_cast<byte>(minimum), static_cast<byte>(maximum),
        static_cast<byte>(weights.size())};
    for (int index = 0; index < static_cast<int>(weights.size()); ++index) {
      const std::vector<byte> card = {
          0, 0, 0, 0, static_cast<byte>(player), LOCATION_MZONE,
          static_cast<byte>(index), static_cast<byte>(weights[index])};
      bytes.insert(bytes.end(), card.begin(), card.end());
    }
    std::copy(bytes.begin(), bytes.end(), data_);
    dp_ = 0;
    dl_ = static_cast<int>(bytes.size());
    handle_message();
    require(dp_ == dl_, "Tribute message was not fully consumed");
    require(to_play_ == player, "Wrong selecting player");
  }

  bool has_finish() const {
    return std::any_of(legal_actions_.begin(), legal_actions_.end(),
                       [](const LegalAction& action) { return action.finish_; });
  }

  size_t option_count() const { return legal_actions_.size(); }

  void choose(const std::string& spec) {
    for (int index = 0; index < static_cast<int>(legal_actions_.size()); ++index) {
      const auto& action = legal_actions_[index];
      if ((spec == "finish" && action.finish_) ||
          (!action.finish_ && action.spec_ == spec)) {
        callback_(index);
        if (ms_idx_ != -1) {
          handle_multi_select();
        }
        return;
      }
    }
    throw std::runtime_error("Requested action is not legal: " + spec);
  }

  void expect_response(const std::vector<int>& indices) const {
    require(ms_idx_ == -1, "Selection did not finish");
    require(resp_buf_[0] == indices.size(), "Wrong response card count");
    for (size_t index = 0; index < indices.size(); ++index) {
      require(resp_buf_[index + 1] == indices[index], "Wrong response index");
    }
  }
};

void positive_minimum() {
  {
    TributeFixture fixture;
    fixture.message(1, 2, {1, 2});
    require(!fixture.has_finish(), "Positive minimum allowed empty selection");
    fixture.choose("m1");
    require(fixture.has_finish(), "Satisfied minimum cannot finish");
    fixture.choose("finish");
    fixture.expect_response({0});
  }
  {
    TributeFixture fixture;
    fixture.message(2, 1, {1, 2}, 1);
    require(fixture.option_count() == 1, "Weighted tribute ignored card limit");
    fixture.choose("m2");
    fixture.expect_response({1});
  }
  {
    TributeFixture fixture;
    bool rejected = false;
    try {
      fixture.message(3, 1, {1, 2});
    } catch (const std::runtime_error& error) {
      rejected = std::string(error.what()).find("No valid select tribute") !=
                 std::string::npos;
    }
    require(rejected, "Impossible tribute accepted");
  }
}

void zero_minimum() {
  {
    TributeFixture fixture;
    fixture.message(0, 2, {1, 2});
    require(fixture.option_count() == 3, "Expected two cards and finish");
    fixture.choose("finish");
    fixture.expect_response({});
  }
  {
    TributeFixture fixture;
    fixture.message(0, 2, {1, 2}, 1);
    fixture.choose("m1");
    require(fixture.has_finish(), "Optional tribute lost early finish");
    fixture.choose("finish");
    fixture.expect_response({0});
  }
  {
    TributeFixture fixture;
    fixture.message(0, 2, {1, 2});
    fixture.choose("m1");
    fixture.choose("m2");
    fixture.expect_response({0, 1});
  }
  {
    TributeFixture fixture;
    fixture.message(0, 2, {1, 2});
    fixture.choose("m2");
    fixture.expect_response({1});
  }
  {
    TributeFixture fixture;
    fixture.message(0, 0, {1, 2});
    require(fixture.option_count() == 1, "Zero maximum offered a card");
    fixture.choose("finish");
    fixture.expect_response({});
  }
  {
    TributeFixture fixture;
    fixture.message(0, 0, {});
    require(fixture.option_count() == 1, "Empty candidate set cannot finish");
    fixture.choose("finish");
    fixture.expect_response({});
  }
}

int main(int argc, char** argv) {
  try {
    positive_minimum();
    if (argc == 1 || std::string(argv[1]) != "--positive-only") {
      zero_minimum();
    }
    std::cout << "TRIBUTE_TESTS_PASSED\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "TRIBUTE_TEST_FAILED: " << error.what() << "\n";
    return 1;
  }
}
