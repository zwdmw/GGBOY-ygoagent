#ifndef YGOENV_YGOPRO_YGOPRO_H_
#define YGOENV_YGOPRO_YGOPRO_H_

// clang-format off
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <numeric>
#include <stdexcept>
#include <string>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>


#include <fmt/core.h>
#include <fmt/ranges.h>
#include <SQLiteCpp/SQLiteCpp.h>
#include <SQLiteCpp/VariadicBind.h>
#include <ankerl/unordered_dense.h>
#include <unordered_set>

#include "ygoenv/core/BS_thread_pool.h"

#include "ygoenv/core/async_envpool.h"
#include "ygoenv/core/env.h"

#include "ygopro-core/common.h"
#include "ygopro-core/card_data.h"
#include "ygopro-core/duel.h"
#include "ygopro-core/ocgapi.h"

// clang-format on

namespace ygopro {

inline std::vector<std::vector<int>> combinations(int n, int r) {
  std::vector<std::vector<int>> combs;
  std::vector<bool> m(n);
  std::fill(m.begin(), m.begin() + r, true);

  do {
    std::vector<int> cs;
    cs.reserve(r);
    for (int i = 0; i < n; ++i) {
      if (m[i]) {
        cs.push_back(i);
      }
    }
    combs.push_back(cs);
  } while (std::prev_permutation(m.begin(), m.end()));

  return combs;
}

inline bool sum_to(const std::vector<int> &w, const std::vector<int> ind, int i,
                   int r) {
  if (r <= 0) {
    return false;
  }
  int n = ind.size();
  if (i == n - 1) {
    return r == 1 || (w[ind[i]] == r);
  }
  return sum_to(w, ind, i + 1, r - 1) || sum_to(w, ind, i + 1, r - w[ind[i]]);
}

inline bool sum_to(const std::vector<int> &w, const std::vector<int> ind,
                   int r) {
  return sum_to(w, ind, 0, r);
}

inline std::vector<std::vector<int>>
combinations_with_weight(const std::vector<int> &weights, int r) {
  int n = weights.size();
  std::vector<std::vector<int>> results;

  for (int k = 1; k <= n; k++) {
    std::vector<std::vector<int>> combs = combinations(n, k);
    for (const auto &comb : combs) {
      if (sum_to(weights, comb, r)) {
        results.push_back(comb);
      }
    }
  }
  return results;
}

inline std::vector<std::vector<int>> tribute_combinations(
    const std::vector<int> &weights, int minimum_value,
    int maximum_cards) {
  const int count = static_cast<int>(weights.size());
  const int begin = minimum_value <= 0 ? 0 : 1;
  const int end = std::min(count, std::max(0, maximum_cards));
  std::vector<std::vector<int>> results;

  for (int size = begin; size <= end; ++size) {
    for (const auto &combination : combinations(count, size)) {
      int value = 0;
      for (const auto index : combination) {
        value += weights[index];
      }
      if (value >= minimum_value) {
        results.push_back(combination);
      }
    }
  }
  return results;
}

inline std::vector<int> sum_param_values(uint32_t param) {
  std::vector<int> values;
  const int value1 = param & 0xffff;
  const int value2 = param >> 16;
  if (value1 > 0) {
    values.push_back(value1);
  }
  if (value2 > 0 && value2 != value1) {
    values.push_back(value2);
  }
  return values;
}

inline bool sum_params_exact(
    const std::vector<uint32_t> &params, int index, int remaining) {
  if (index == static_cast<int>(params.size())) {
    return remaining == 0;
  }
  if (remaining <= 0) {
    return false;
  }
  for (const auto value : sum_param_values(params[index])) {
    if (value <= remaining &&
        sum_params_exact(params, index + 1, remaining - value)) {
      return true;
    }
  }
  return false;
}

inline bool sum_params_exact(
    const std::vector<uint32_t> &params, int target) {
  return !params.empty() && sum_params_exact(params, 0, target);
}

inline bool sum_params_at_least_minimal(
    const std::vector<uint32_t> &params, int target) {
  if (params.empty()) {
    return false;
  }
  int64_t minimum_sum = 0;
  int64_t maximum_sum = 0;
  int minimum_value = 0x7fffffff;
  for (const auto param : params) {
    const int value1 = param & 0xffff;
    const int value2 = param >> 16;
    const int current_minimum =
        value2 > 0 && value2 < value1 ? value2 : value1;
    minimum_sum += current_minimum;
    maximum_sum += std::max(value1, value2);
    minimum_value = std::min(minimum_value, current_minimum);
  }
  return maximum_sum >= target &&
         minimum_sum - minimum_value < target;
}

inline std::vector<std::vector<int>> combinations_with_sum_params(
    const std::vector<uint32_t> &must_params,
    const std::vector<uint32_t> &select_params, int target,
    int minimum_count, int maximum_count, bool at_least) {
  const int count = static_cast<int>(select_params.size());
  const int begin = at_least ? 0 : std::max(0, minimum_count);
  const int end = at_least
      ? count
      : std::min(count, std::max(minimum_count, maximum_count));
  std::vector<std::vector<int>> results;

  for (int size = begin; size <= end; ++size) {
    for (const auto &combination : combinations(count, size)) {
      std::vector<uint32_t> params = must_params;
      params.reserve(must_params.size() + combination.size());
      for (const auto index : combination) {
        params.push_back(select_params[index]);
      }
      const bool valid = at_least
          ? sum_params_at_least_minimal(params, target)
          : sum_params_exact(params, target);
      if (valid) {
        results.push_back(combination);
      }
    }
  }
  return results;
}

inline bool evaluate_announce_card_filter(
    const card_data &card, const std::vector<uint32_t> &opcodes) {
  std::vector<int64_t> stack;
  for (const auto opcode : opcodes) {
    auto pop_binary = [&stack]() {
      const auto rhs = stack.back();
      stack.pop_back();
      const auto lhs = stack.back();
      stack.pop_back();
      return std::pair<int64_t, int64_t>{lhs, rhs};
    };
    switch (opcode) {
    case OPCODE_ADD:
      if (stack.size() >= 2) {
        const auto [lhs, rhs] = pop_binary();
        stack.push_back(lhs + rhs);
      }
      break;
    case OPCODE_SUB:
      if (stack.size() >= 2) {
        const auto [lhs, rhs] = pop_binary();
        stack.push_back(lhs - rhs);
      }
      break;
    case OPCODE_MUL:
      if (stack.size() >= 2) {
        const auto [lhs, rhs] = pop_binary();
        stack.push_back(lhs * rhs);
      }
      break;
    case OPCODE_DIV:
      if (stack.size() >= 2) {
        const auto [lhs, rhs] = pop_binary();
        if (rhs == 0) {
          return false;
        }
        stack.push_back(lhs / rhs);
      }
      break;
    case OPCODE_AND:
      if (stack.size() >= 2) {
        const auto [lhs, rhs] = pop_binary();
        stack.push_back(lhs && rhs);
      }
      break;
    case OPCODE_OR:
      if (stack.size() >= 2) {
        const auto [lhs, rhs] = pop_binary();
        stack.push_back(lhs || rhs);
      }
      break;
    case OPCODE_NEG:
      if (!stack.empty()) {
        stack.back() = -stack.back();
      }
      break;
    case OPCODE_NOT:
      if (!stack.empty()) {
        stack.back() = !stack.back();
      }
      break;
    case OPCODE_ISCODE:
      if (!stack.empty()) {
        stack.back() = card.code == static_cast<uint32_t>(stack.back());
      }
      break;
    case OPCODE_ISSETCARD:
      if (!stack.empty()) {
        stack.back() = card.is_setcode(static_cast<uint32_t>(stack.back()));
      }
      break;
    case OPCODE_ISTYPE:
      if (!stack.empty()) {
        stack.back() = card.type & static_cast<uint32_t>(stack.back());
      }
      break;
    case OPCODE_ISRACE:
      if (!stack.empty()) {
        stack.back() = card.race & static_cast<uint32_t>(stack.back());
      }
      break;
    case OPCODE_ISATTRIBUTE:
      if (!stack.empty()) {
        stack.back() = card.attribute & static_cast<uint32_t>(stack.back());
      }
      break;
    default:
      stack.push_back(static_cast<int32_t>(opcode));
      break;
    }
  }
  if (stack.size() != 1 || stack.back() == 0) {
    return false;
  }
  return card.code == 78734254u || card.code == 13857930u ||
         (!card.alias &&
          (card.type & (TYPE_MONSTER | TYPE_TOKEN)) !=
              (TYPE_MONSTER | TYPE_TOKEN));
}

static std::string msg_to_string(int msg) {
  switch (msg) {
  case MSG_RETRY:
    return "retry";
  case MSG_HINT:
    return "hint";
  case MSG_WIN:
    return "win";
  case MSG_SELECT_BATTLECMD:
    return "select_battlecmd";
  case MSG_SELECT_IDLECMD:
    return "select_idlecmd";
  case MSG_SELECT_EFFECTYN:
    return "select_effectyn";
  case MSG_SELECT_YESNO:
    return "select_yesno";
  case MSG_SELECT_OPTION:
    return "select_option";
  case MSG_SELECT_CARD:
    return "select_card";
  case MSG_SELECT_CHAIN:
    return "select_chain";
  case MSG_SELECT_PLACE:
    return "select_place";
  case MSG_SELECT_POSITION:
    return "select_position";
  case MSG_SELECT_TRIBUTE:
    return "select_tribute";
  case MSG_SELECT_COUNTER:
    return "select_counter";
  case MSG_SELECT_SUM:
    return "select_sum";
  case MSG_SELECT_DISFIELD:
    return "select_disfield";
  case MSG_SORT_CARD:
    return "sort_card";
  case MSG_SELECT_UNSELECT_CARD:
    return "select_unselect_card";
  case MSG_CONFIRM_DECKTOP:
    return "confirm_decktop";
  case MSG_CONFIRM_CARDS:
    return "confirm_cards";
  case MSG_SHUFFLE_DECK:
    return "shuffle_deck";
  case MSG_SHUFFLE_HAND:
    return "shuffle_hand";
  case MSG_SWAP_GRAVE_DECK:
    return "swap_grave_deck";
  case MSG_SHUFFLE_SET_CARD:
    return "shuffle_set_card";
  case MSG_REVERSE_DECK:
    return "reverse_deck";
  case MSG_DECK_TOP:
    return "deck_top";
  case MSG_SHUFFLE_EXTRA:
    return "shuffle_extra";
  case MSG_NEW_TURN:
    return "new_turn";
  case MSG_NEW_PHASE:
    return "new_phase";
  case MSG_CONFIRM_EXTRATOP:
    return "confirm_extratop";
  case MSG_MOVE:
    return "move";
  case MSG_POS_CHANGE:
    return "pos_change";
  case MSG_SET:
    return "set";
  case MSG_SWAP:
    return "swap";
  case MSG_FIELD_DISABLED:
    return "field_disabled";
  case MSG_SUMMONING:
    return "summoning";
  case MSG_SUMMONED:
    return "summoned";
  case MSG_SPSUMMONING:
    return "spsummoning";
  case MSG_SPSUMMONED:
    return "spsummoned";
  case MSG_FLIPSUMMONING:
    return "flipsummoning";
  case MSG_FLIPSUMMONED:
    return "flipsummoned";
  case MSG_CHAINING:
    return "chaining";
  case MSG_CHAINED:
    return "chained";
  case MSG_CHAIN_SOLVING:
    return "chain_solving";
  case MSG_CHAIN_SOLVED:
    return "chain_solved";
  case MSG_CHAIN_END:
    return "chain_end";
  case MSG_CHAIN_NEGATED:
    return "chain_negated";
  case MSG_CHAIN_DISABLED:
    return "chain_disabled";
  case MSG_RANDOM_SELECTED:
    return "random_selected";
  case MSG_BECOME_TARGET:
    return "become_target";
  case MSG_DRAW:
    return "draw";
  case MSG_DAMAGE:
    return "damage";
  case MSG_RECOVER:
    return "recover";
  case MSG_EQUIP:
    return "equip";
  case MSG_LPUPDATE:
    return "lpupdate";
  case MSG_CARD_TARGET:
    return "card_target";
  case MSG_CANCEL_TARGET:
    return "cancel_target";
  case MSG_PAY_LPCOST:
    return "pay_lpcost";
  case MSG_ADD_COUNTER:
    return "add_counter";
  case MSG_REMOVE_COUNTER:
    return "remove_counter";
  case MSG_ATTACK:
    return "attack";
  case MSG_BATTLE:
    return "battle";
  case MSG_ATTACK_DISABLED:
    return "attack_disabled";
  case MSG_DAMAGE_STEP_START:
    return "damage_step_start";
  case MSG_DAMAGE_STEP_END:
    return "damage_step_end";
  case MSG_MISSED_EFFECT:
    return "missed_effect";
  case MSG_TOSS_COIN:
    return "toss_coin";
  case MSG_TOSS_DICE:
    return "toss_dice";
  case MSG_ROCK_PAPER_SCISSORS:
    return "rock_paper_scissors";
  case MSG_HAND_RES:
    return "hand_res";
  case MSG_ANNOUNCE_RACE:
    return "announce_race";
  case MSG_ANNOUNCE_ATTRIB:
    return "announce_attrib";
  case MSG_ANNOUNCE_CARD:
    return "announce_card";
  case MSG_ANNOUNCE_NUMBER:
    return "announce_number";
  case MSG_CARD_HINT:
    return "card_hint";
  case MSG_TAG_SWAP:
    return "tag_swap";
  case MSG_RELOAD_FIELD:
    return "reload_field";
  case MSG_AI_NAME:
    return "ai_name";
  case MSG_SHOW_HINT:
    return "show_hint";
  case MSG_PLAYER_HINT:
    return "player_hint";
  case MSG_MATCH_KILL:
    return "match_kill";
  case MSG_CUSTOM_MSG:
    return "custom_msg";
  default:
    return "unknown_msg";
  }
}

// system string
static const std::map<int, std::string> system_strings = {
    // announce type
    {1050, "Monster"},
    {1051, "Spell"},
    {1052, "Trap"},
    {1054, "Normal"},
    {1055, "Effect"},
    {1056, "Fusion"},
    {1057, "Ritual"},
    {1058, "Trap Monsters"},
    {1059, "Spirit"},
    {1060, "Union"},
    {1061, "Gemini"},
    {1062, "Tuner"},
    {1063, "Synchro"},
    {1064, "Token"},
    {1066, "Quick-Play"},
    {1067, "Continuous"},
    {1068, "Equip"},
    {1069, "Field"},
    {1070, "Counter"},
    {1071, "Flip"},
    {1072, "Toon"},
    {1073, "Xyz"},
    {1074, "Pendulum"},
    {1075, "Special Summon"},
    {1076, "Link"},
    {1080, "(N/A)"},
    {1081, "Extra Monster Zone"},
    // announce type end
    // actions
    {1150, "Activate"},
    {1151, "Normal Summon"},
    {1152, "Special Summon"},
    {1153, "Set"},
    {1154, "Flip Summon"},
    {1155, "To Defense"},
    {1156, "To Attack"},
    {1157, "Attack"},
    {1158, "View"},
    {1159, "S/T Set"},
    {1160, "Put in Pendulum Zone"},
    {1161, "Do Effect"},
    {1162, "Reset Effect"},
    {1163, "Pendulum Summon"},
    {1164, "Synchro Summon"},
    {1165, "Xyz Summon"},
    {1166, "Link Summon"},
    {1167, "Tribute Summon"},
    {1168, "Ritual Summon"},
    {1169, "Fusion Summon"},
    {1190, "Add to hand"},
    {1191, "Send to GY"},
    {1192, "Banish"},
    {1193, "Return to Deck"},
    // actions end
    {1, "Normal Summon"},
    {30, "Replay rules apply. Continue this attack?"},
    {31, "Attack directly with this monster?"},
    {80, "Start Step of the Battle Phase."},
    {81, "During the End Phase."},
    {90, "Conduct this Normal Summon without Tributing?"},
    {91, "Use additional Summon?"},
    {92, "Tribute your opponent's monster?"},
    {93, "Continue selecting Materials?"},
    {94, "Activate this card's effect now?"},
    {95, "Use the effect of [%ls]?"},
    {96, "Use the effect of [%ls] to avoid destruction?"},
    {97, "Place [%ls] to a Spell & Trap Zone?"},
    {98, "Tribute a monster(s) your opponent controls?"},
    {200, "From [%ls], activate [%ls]?"},
    {203, "Chain another card or effect?"},
    {210, "Continue selecting?"},
    {218, "Pay LP by Effect of [%ls], instead?"},
    {219, "Detach Xyz material by Effect of [%ls], instead?"},
    {220, "Remove Counter(s) by Effect of [%ls], instead?"},
    {221, "On [%ls], Activate Trigger Effect of [%ls]?"},
    {222, "Activate Trigger Effect?"},
    {221, "On [%ls], Activate Trigger Effect of [%ls]?"},
    {1621, "Attack Negated"},
    {1622, "[%ls] Missed timing"}
};

// Keep the legacy system-string IDs stable for checkpoint compatibility.
// New script-level choices are assigned after the legacy range.
static const std::map<int, std::string> system_string_extensions = {
    {1213, "Yes"},
    {1214, "No"},
};

// Standard descriptions used directly by this script snapshot. Keep these in
// a separate map so adding lower numeric keys cannot renumber 1213/1214.
static const std::map<int, std::string> script_system_string_extensions = {
    {10, "Remove Counter(s)"},
    {66, "Keep revealed"},
    {1104, "Back to Hand"},
    {1623, "Coin landed on:"},
};

// Generic engine prompts below 1000 fit in the remaining byte range and are
// appended after every checkpoint-compatible ID above.
static const std::vector<int> generic_system_string_extensions = {
    2,   3,   4,   5,   6,   7,   11,  12,  20,  21,  22,  23,
    24,  25,  26,  27,  28,  29,  40,  41,  42,  43,  44,  60,
    61,  62,  63,  64,  65,  67,  70,  71,  72,  100, 101, 102,
    103, 201, 202, 204, 205, 206, 207, 208, 209, 211, 212, 213,
    214, 215, 216, 217, 223, 224, 225, 500, 501, 502, 503, 504,
    505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516,
    517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528,
    529, 530, 531, 532, 533, 534, 549, 550, 551, 552, 553, 554,
    555, 556, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569,
    570, 571, 572, 573, 574, 575,
};

static std::string get_system_string(int desc) {
  auto it = system_strings.find(desc);
  if (it != system_strings.end()) {
    return it->second;
  }
  auto extension_it = system_string_extensions.find(desc);
  if (extension_it != system_string_extensions.end()) {
    return extension_it->second;
  }
  auto script_extension_it = script_system_string_extensions.find(desc);
  if (script_extension_it != script_system_string_extensions.end()) {
    return script_extension_it->second;
  }
  return "system string " + std::to_string(desc);
}

static std::string ltrim(std::string s) {
  s.erase(s.begin(),
          std::find_if(s.begin(), s.end(),
                       std::not1(std::ptr_fun<int, int>(std::isspace))));
  return s;
}


inline std::string ls_to_spec(uint8_t loc, uint8_t seq, uint8_t pos) {
  std::string spec;
  if (loc & LOCATION_HAND) {
    spec += "h";
  } else if (loc & LOCATION_MZONE) {
    spec += "m";
  } else if (loc & LOCATION_SZONE) {
    spec += "s";
  } else if (loc & LOCATION_GRAVE) {
    spec += "g";
  } else if (loc & LOCATION_REMOVED) {
    spec += "r";
  } else if (loc & LOCATION_EXTRA) {
    spec += "x";
  }
  spec += std::to_string(seq + 1);
  if (loc & LOCATION_OVERLAY) {
    spec.push_back('a' + pos);
  }
  return spec;
}

inline std::string ls_to_spec(uint8_t loc, uint8_t seq, uint8_t pos, bool opponent) {
  std::string spec = ls_to_spec(loc, seq, pos);
  if (opponent) {
    spec.insert(0, 1, 'o');
  }
  return spec;
}

inline std::tuple<uint8_t, uint8_t, uint8_t>
spec_to_ls(const std::string spec) {
  uint8_t loc;
  uint8_t seq;
  uint8_t pos = 0;
  int offset = 1;
  if (spec[0] == 'h') {
    loc = LOCATION_HAND;
  } else if (spec[0] == 'm') {
    loc = LOCATION_MZONE;
  } else if (spec[0] == 's') {
    loc = LOCATION_SZONE;
  } else if (spec[0] == 'g') {
    loc = LOCATION_GRAVE;
  } else if (spec[0] == 'r') {
    loc = LOCATION_REMOVED;
  } else if (spec[0] == 'x') {
    loc = LOCATION_EXTRA;
  } else if (std::isdigit(spec[0])) {
    loc = LOCATION_DECK;
    offset = 0;
  } else {
    std::string s = fmt::format("Invalid spec {}", spec);
    throw std::runtime_error(s);
  }
  int end = offset;
  while (end < spec.size() && std::isdigit(spec[end])) {
    end++;
  }
  seq = std::stoi(spec.substr(offset, end - offset)) - 1;
  if (end < spec.size()) {
    pos = spec[end] - 'a';
  }
  return {loc, seq, pos};
}


inline std::tuple<uint8_t, uint8_t, uint8_t, uint8_t>
spec_to_ls(uint8_t player, const std::string spec) {
  uint8_t controller = player;
  int offset = 0;
  if (spec[0] == 'o') {
    controller = 1 - player;
    offset++;
  }
  auto [loc, seq, pos] = spec_to_ls(spec.substr(offset));
  return {controller, loc, seq, pos};
}


static std::tuple<std::vector<uint32>, std::vector<uint32>, std::vector<uint32>> read_decks(const std::string &fp) {
  std::ifstream file(fp);
  std::string line;
  std::vector<uint32> main_deck, extra_deck, side_deck;
  bool found_extra = false;

  if (file.is_open()) {
    // Read the main deck
    while (std::getline(file, line)) {
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }
      if (line.find("side") != std::string::npos) {
        break;
      }
      if (line.find("extra") != std::string::npos) {
        found_extra = true;
        break;
      }
      // Check if line contains only digits
      if (std::all_of(line.begin(), line.end(), ::isdigit)) {
        main_deck.push_back(std::stoul(line));
      }
    }

    if (main_deck.size() < 40) {
      std::string err = fmt::format("Main deck must contain at least 40 cards, found: {}, file: {}", main_deck.size(), fp);
      throw std::runtime_error(err);
    }

    // Read the extra deck
    if (found_extra) {
      while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
          line.pop_back();
        }
        if (line.find("side") != std::string::npos) {
          break;
        }
        // Check if line contains only digits
        if (std::all_of(line.begin(), line.end(), ::isdigit)) {
          extra_deck.push_back(std::stoul(line));
        }
      }
    }

    // Read the side deck
    while (std::getline(file, line)) {
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }
      // Check if line contains only digits
      if (std::all_of(line.begin(), line.end(), ::isdigit)) {
        side_deck.push_back(std::stoul(line));
      }
    }

    file.close();
  } else {
    throw std::runtime_error(fmt::format("Unable to open deck file: {}", fp));
  }

  return std::make_tuple(main_deck, extra_deck, side_deck);
}

template <class K = uint8_t>
ankerl::unordered_dense::map<K, uint8_t>
make_ids(const std::map<K, std::string> &m, int id_offset = 0,
         int m_offset = 0) {
  ankerl::unordered_dense::map<K, uint8_t> m2;
  int i = 0;
  for (const auto &[k, v] : m) {
    if (i < m_offset) {
      i++;
      continue;
    }
    m2[k] = i - m_offset + id_offset;
    i++;
  }
  return m2;
}

template <class K = char>
ankerl::unordered_dense::map<K, uint8_t>
make_ids(const std::vector<K> &cmds, int id_offset = 0, int m_offset = 0) {
  ankerl::unordered_dense::map<K, uint8_t> m2;
  for (int i = m_offset; i < cmds.size(); i++) {
    m2[cmds[i]] = i - m_offset + id_offset;
  }
  return m2;
}

static std::string reason_to_string(uint8_t reason) {
  // !victory 0x0 Surrendered
  // !victory 0x1 LP reached 0
  // !victory 0x2 Cards can't be drawn
  // !victory 0x3 Time limit up
  // !victory 0x4 Lost connection
  switch (reason) {
  case 0x0:
    return "Surrendered";
  case 0x1:
    return "LP reached 0";
  case 0x2:
    return "Cards can't be drawn";
  case 0x3:
    return "Time limit up";
  case 0x4:
    return "Lost connection";
  default:
    return "Unknown";
  }
}

#define DEFINE_X_TO_ID_FUN(name, x_map) \
inline uint8_t name(decltype(x_map)::key_type x) { \
  auto it = x_map.find(x); \
  if (it != x_map.end()) { \
    return it->second; \
  } \
  throw std::runtime_error( \
    fmt::format("[" #name "] cannot find id: {}", x)); \
}

#define DEFINE_X_TO_STRING_FUN(name, x_map) \
inline std::string name(decltype(x_map)::key_type x) { \
  auto it = x_map.find(x); \
  if (it != x_map.end()) { \
    return it->second; \
  } \
  return "unknown"; \
}

static ankerl::unordered_dense::map<int, uint8_t> make_system_string_ids() {
  auto ids = make_ids(system_strings, 16);
  int next_id = 16 + static_cast<int>(system_strings.size());
  for (const auto &entry : system_string_extensions) {
    ids[entry.first] = static_cast<uint8_t>(next_id++);
  }
  for (const auto &entry : script_system_string_extensions) {
    ids[entry.first] = static_cast<uint8_t>(next_id++);
  }
  for (const auto desc : generic_system_string_extensions) {
    if (next_id >= 255) {
      throw std::runtime_error("System string feature IDs exceed byte capacity");
    }
    ids[desc] = static_cast<uint8_t>(next_id++);
  }
  return ids;
}

static const ankerl::unordered_dense::map<int, uint8_t> system_string2id =
    make_system_string_ids();
inline uint8_t system_string_to_id(int desc) {
  auto it = system_string2id.find(desc);
  if (it != system_string2id.end()) {
    return it->second;
  }
  return 255;
}


static const std::map<uint8_t, std::string> location2str = {
    {LOCATION_DECK, "Deck"},
    {LOCATION_HAND, "Hand"},
    {LOCATION_MZONE, "Main Monster Zone"},
    {LOCATION_SZONE, "Spell & Trap Zone"},
    {LOCATION_GRAVE, "Graveyard"},
    {LOCATION_REMOVED, "Banished"},
    {LOCATION_EXTRA, "Extra Deck"},
};

static const ankerl::unordered_dense::map<uint8_t, uint8_t> location2id =
    make_ids(location2str, 1);
DEFINE_X_TO_ID_FUN(location_to_id, location2id)


#define POS_NONE 0x0 // xyz materials (overlay) ???

static const std::map<uint8_t, std::string> position2str = {
    {POS_NONE, "none"},
    {POS_FACEUP_ATTACK, "face-up attack"},
    {POS_FACEDOWN_ATTACK, "face-down attack"},
    {POS_ATTACK, "attack"},
    {POS_FACEUP_DEFENSE, "face-up defense"},
    {POS_FACEUP, "face-up"},
    {POS_FACEDOWN_DEFENSE, "face-down defense"},
    {POS_FACEDOWN, "face-down"},
    {POS_DEFENSE, "defense"},
};
DEFINE_X_TO_STRING_FUN(position_to_string, position2str)

static const ankerl::unordered_dense::map<uint8_t, uint8_t> position2id =
    make_ids(position2str);
DEFINE_X_TO_ID_FUN(position_to_id, position2id)


#define ATTRIBUTE_NONE 0x0 // token

static const std::map<uint8_t, std::string> attribute2str = {
    {ATTRIBUTE_NONE, "None"},   {ATTRIBUTE_EARTH, "Earth"},
    {ATTRIBUTE_WATER, "Water"}, {ATTRIBUTE_FIRE, "Fire"},
    {ATTRIBUTE_WIND, "Wind"},   {ATTRIBUTE_LIGHT, "Light"},
    {ATTRIBUTE_DARK, "Dark"},   {ATTRIBUTE_DEVINE, "Divine"},
};
DEFINE_X_TO_STRING_FUN(attribute_to_string, attribute2str)

static const ankerl::unordered_dense::map<uint8_t, uint8_t> attribute2id =
    make_ids(attribute2str);
DEFINE_X_TO_ID_FUN(attribute_to_id, attribute2id)


#define RACE_NONE 0x0 // token

static const std::map<uint32_t, std::string> race2str = {
    {RACE_NONE, "None"},
    {RACE_WARRIOR, "Warrior"},
    {RACE_SPELLCASTER, "Spellcaster"},
    {RACE_FAIRY, "Fairy"},
    {RACE_FIEND, "Fiend"},
    {RACE_ZOMBIE, "Zombie"},
    {RACE_MACHINE, "Machine"},
    {RACE_AQUA, "Aqua"},
    {RACE_PYRO, "Pyro"},
    {RACE_ROCK, "Rock"},
    {RACE_WINDBEAST, "Windbeast"},
    {RACE_PLANT, "Plant"},
    {RACE_INSECT, "Insect"},
    {RACE_THUNDER, "Thunder"},
    {RACE_DRAGON, "Dragon"},
    {RACE_BEAST, "Beast"},
    {RACE_BEASTWARRIOR, "Beast Warrior"},
    {RACE_DINOSAUR, "Dinosaur"},
    {RACE_FISH, "Fish"},
    {RACE_SEASERPENT, "Sea Serpent"},
    {RACE_REPTILE, "Reptile"},
    {RACE_PSYCHO, "Psycho"},
    {RACE_DEVINE, "Divine"},
    {RACE_CREATORGOD, "Creator God"},
    {RACE_WYRM, "Wyrm"},
    {RACE_CYBERSE, "Cyberse"},
    {RACE_ILLUSION, "Illusion"}};

static const ankerl::unordered_dense::map<uint32_t, uint8_t> race2id =
    make_ids(race2str);
DEFINE_X_TO_ID_FUN(race_to_id, race2id)


static const std::map<uint32_t, std::string> type2str = {
    {TYPE_MONSTER, "Monster"},
    {TYPE_SPELL, "Spell"},
    {TYPE_TRAP, "Trap"},
    {TYPE_NORMAL, "Normal"},
    {TYPE_EFFECT, "Effect"},
    {TYPE_FUSION, "Fusion"},
    {TYPE_RITUAL, "Ritual"},
    {TYPE_TRAPMONSTER, "Trap Monster"},
    {TYPE_SPIRIT, "Spirit"},
    {TYPE_UNION, "Union"},
    {TYPE_DUAL, "Dual"},
    {TYPE_TUNER, "Tuner"},
    {TYPE_SYNCHRO, "Synchro"},
    {TYPE_TOKEN, "Token"},
    {TYPE_QUICKPLAY, "Quick-play"},
    {TYPE_CONTINUOUS, "Continuous"},
    {TYPE_EQUIP, "Equip"},
    {TYPE_FIELD, "Field"},
    {TYPE_COUNTER, "Counter"},
    {TYPE_FLIP, "Flip"},
    {TYPE_TOON, "Toon"},
    {TYPE_XYZ, "XYZ"},
    {TYPE_PENDULUM, "Pendulum"},
    {TYPE_SPSUMMON, "Special"},
    {TYPE_LINK, "Link"},
};

inline std::vector<uint8_t> type_to_ids(uint32_t type) {
  std::vector<uint8_t> ids;
  ids.reserve(type2str.size());
  for (const auto &[k, v] : type2str) {
    ids.push_back(std::min(1u, type & k));
  }
  return ids;
}

static const std::map<int, std::string> phase2str = {
    {PHASE_DRAW, "draw phase"},
    {PHASE_STANDBY, "standby phase"},
    {PHASE_MAIN1, "main1 phase"},
    {PHASE_BATTLE_START, "battle start phase"},
    {PHASE_BATTLE_STEP, "battle step phase"},
    {PHASE_DAMAGE, "damage phase"},
    {PHASE_DAMAGE_CAL, "damage calculation phase"},
    {PHASE_BATTLE, "battle phase"},
    {PHASE_MAIN2, "main2 phase"},
    {PHASE_END, "end phase"},
};
DEFINE_X_TO_STRING_FUN(phase_to_string, phase2str)

static const ankerl::unordered_dense::map<int, uint8_t> phase2id =
    make_ids(phase2str);
DEFINE_X_TO_ID_FUN(phase_to_id, phase2id)


static const std::vector<int> _msgs = {
    MSG_SELECT_IDLECMD,  MSG_SELECT_CHAIN,     MSG_SELECT_CARD,
    MSG_SELECT_TRIBUTE,  MSG_SELECT_POSITION,  MSG_SELECT_EFFECTYN,
    MSG_SELECT_YESNO,    MSG_SELECT_BATTLECMD, MSG_SELECT_UNSELECT_CARD,
    MSG_SELECT_OPTION,   MSG_SELECT_PLACE,     MSG_SELECT_SUM,
    MSG_SELECT_DISFIELD, MSG_ANNOUNCE_ATTRIB,  MSG_ANNOUNCE_NUMBER,
    MSG_ANNOUNCE_CARD,   MSG_ANNOUNCE_RACE,    MSG_TOSS_COIN,
    MSG_TOSS_DICE,       MSG_SWAP_GRAVE_DECK,
};

static const ankerl::unordered_dense::map<int, uint8_t> msg2id =
    make_ids(_msgs, 1);
DEFINE_X_TO_ID_FUN(msg_to_id, msg2id)


enum class ActionAct {
  None,
  Set,
  Repo,
  SpSummon,
  Summon,
  MSet,
  Attack,
  DirectAttack,
  Activate,
  Cancel,
};

inline std::string action_act_to_string(ActionAct act) {
  switch (act) {
  case ActionAct::None:
    return "None";
  case ActionAct::Set:
    return "Set";
  case ActionAct::Repo:
    return "Repo";
  case ActionAct::SpSummon:
    return "SpSummon";
  case ActionAct::Summon:
    return "Summon";
  case ActionAct::MSet:
    return "MSet";
  case ActionAct::Attack:
    return "Attack";
  case ActionAct::DirectAttack:
    return "DirectAttack";
  case ActionAct::Activate:
    return "Activate";
  case ActionAct::Cancel:
    return "Cancel";
  default:
    return "Unknown";
  }
}

enum class ActionPhase {
  None,
  Battle,
  Main2,
  End,
};

inline std::string action_phase_to_string(ActionPhase phase) {
  switch (phase) {
  case ActionPhase::None:
    return "None";
  case ActionPhase::Battle:
    return "Battle";
  case ActionPhase::Main2:
    return "Main2";
  case ActionPhase::End:
    return "End";
  default:
    return "Unknown";
  }
}

enum class ActionPlace {
  None,
  MZone1,
  MZone2,
  MZone3,
  MZone4,
  MZone5,
  MZone6,
  MZone7,
  SZone1,
  SZone2,
  SZone3,
  SZone4,
  SZone5,
  SZone6,
  SZone7,
  SZone8,
  OpMZone1,
  OpMZone2,
  OpMZone3,
  OpMZone4,
  OpMZone5,
  OpMZone6,
  OpMZone7,
  OpSZone1,
  OpSZone2,
  OpSZone3,
  OpSZone4,
  OpSZone5,
  OpSZone6,
  OpSZone7,
  OpSZone8,
};

enum class SelectionRole : uint8_t {
  None = 0,
  Candidate = 1,
  Target = 2,
  Cost = 3,
  Material = 4,
  Tribute = 5,
};

constexpr int kActionIRFeatures = 24;
constexpr int kActionSingleRefs = 4;
constexpr int kActionGroupRoles = 5;
constexpr int kMaxActionRoleMembers = 8;
constexpr int kSelectionFeatures = 12;
constexpr int kPublicEventFeatures = 16;
constexpr int kPublicEventRefs = 4;


inline std::vector<ActionPlace> flag_to_usable_places(
  uint32_t flag, bool reverse = false) {
  std::vector<ActionPlace> places;
  for (int j = 0; j < 4; j++) {
    uint32_t value = (flag >> (j * 8)) & 0xff;
    for (int i = 0; i < 8; i++) {
      bool avail = (value & (1 << i)) == 0;
      if (reverse) {
        avail = !avail;
      }
      if (avail) {
        ActionPlace place;
        if (j == 0) {
          place = static_cast<ActionPlace>(i + static_cast<int>(ActionPlace::MZone1));
        } else if (j == 1) {
          place = static_cast<ActionPlace>(i + static_cast<int>(ActionPlace::SZone1));
        } else if (j == 2) {
          place = static_cast<ActionPlace>(i + static_cast<int>(ActionPlace::OpMZone1));
        } else if (j == 3) {
          place = static_cast<ActionPlace>(i + static_cast<int>(ActionPlace::OpSZone1));
        }
        places.push_back(place);
      }
    }
  }
  return places;
}

inline std::string action_place_to_string(ActionPlace place) {
  int i = static_cast<int>(place);
  if (i == 0) {
    return "None";
  }
  else if (i >= static_cast<int>(ActionPlace::MZone1) && i <= static_cast<int>(ActionPlace::MZone7)) {
    return fmt::format("m{}", i - static_cast<int>(ActionPlace::MZone1) + 1);
  }
  else if (i >= static_cast<int>(ActionPlace::SZone1) && i <= static_cast<int>(ActionPlace::SZone8)) {
    return fmt::format("s{}", i - static_cast<int>(ActionPlace::SZone1) + 1);
  }
  else if (i >= static_cast<int>(ActionPlace::OpMZone1) && i <= static_cast<int>(ActionPlace::OpMZone7)) {
    return fmt::format("om{}", i - static_cast<int>(ActionPlace::OpMZone1) + 1);
  }
  else if (i >= static_cast<int>(ActionPlace::OpSZone1) && i <= static_cast<int>(ActionPlace::OpSZone8)) {
    return fmt::format("os{}", i - static_cast<int>(ActionPlace::OpSZone1) + 1);
  }
  else {
    return "Unknown";
  }
}


inline std::pair<uint8_t, uint8_t> float_transform(int x) {
  x = x % 65536;
  return {
      static_cast<uint8_t>(x >> 8),
      static_cast<uint8_t>(x & 0xff),
  };
}

static std::vector<int> find_substrs(const std::string &str,
                                     const std::string &substr) {
  std::vector<int> res;
  int pos = 0;
  while ((pos = str.find(substr, pos)) != std::string::npos) {
    res.push_back(pos);
    pos += substr.length();
  }
  return res;
}

inline std::string time_now() {
  // strftime %Y-%m-%d %H-%M-%S
  time_t now = time(0);
  tm *ltm = localtime(&now);
  char buffer[80];
  strftime(buffer, 80, "%Y-%m-%d %H-%M-%S", ltm);
  return std::string(buffer);
}

// from ygopro/gframe/replay.h

// replay flag
#define REPLAY_COMPRESSED	0x1
#define REPLAY_TAG			0x2
#define REPLAY_DECODED		0x4
#define REPLAY_SINGLE_MODE	0x8
#define REPLAY_UNIFORM		0x10

// max size
#define MAX_REPLAY_SIZE	0x20000


struct ReplayHeader {
	unsigned int id;
	unsigned int version;
	unsigned int flag;
	unsigned int seed;
	unsigned int datasize;
	unsigned int start_time;
	unsigned char props[8];

	ReplayHeader()
		: id(0), version(0), flag(0), seed(0), datasize(0), start_time(0), props{ 0 } {}
};

// from ygopro/gframe/replay.h

using PlayerId = uint8_t;
using CardCode = uint32_t;
using CardId = uint16_t;

const int DESCRIPTION_LIMIT = 10000;
const int CARD_EFFECT_OFFSET = 10010;
const uint8_t ACTION_RACE_EFFECT_OFFSET = 128;
const uint8_t ACTION_ANNOUNCE_NUMBER_MAX = 12;

inline uint8_t announce_number_to_id(uint32_t number) {
  return static_cast<uint8_t>(
      std::min<uint32_t>(number, ACTION_ANNOUNCE_NUMBER_MAX));
}

class LegalAction {
public:
  std::string spec_ = "";
  ActionAct act_ = ActionAct::None;
  ActionPhase phase_ = ActionPhase::None;
  bool finish_ = false;
  uint8_t position_ = 0;
  int effect_ = -1;
  uint8_t number_ = 0;
  ActionPlace place_ = ActionPlace::None;
  uint8_t attribute_ = 0;

  int spec_index_ = 0;
  CardId cid_ = 0;
  int msg_ = 0;
  uint32_t response_ = 0;

  static LegalAction from_spec(const std::string &spec) {
    LegalAction la;
    la.spec_ = spec;
    return la;
  }

  static LegalAction act_spec(ActionAct act, const std::string &spec) {
    LegalAction la;
    la.act_ = act;
    la.spec_ = spec;
    return la;
  }

  static LegalAction finish() {
    LegalAction la;
    la.finish_ = true;
    return la;
  }

  static LegalAction cancel() {
    LegalAction la;
    la.act_ = ActionAct::Cancel;
    return la;
  }

  static LegalAction activate_spec(int effect_idx, const std::string &spec) {
    LegalAction la;
    la.act_ = ActionAct::Activate;
    la.effect_ = effect_idx;
    la.spec_ = spec;
    return la;
  }

  static LegalAction phase(ActionPhase phase) {
    LegalAction la;
    la.phase_ = phase;
    return la;
  }

  static LegalAction number(uint8_t number) {
    LegalAction la;
    la.number_ = number;
    return la;
  }

  static LegalAction place(ActionPlace place) {
    LegalAction la;
    la.place_ = place;
    return la;
  }

  static LegalAction attribute(int attribute) {
    LegalAction la;
    la.attribute_ = attribute;
    return la;
  }
};

struct PublicEvent {
  PlayerId actor = 0;
  int msg = 0;
  ActionAct act = ActionAct::None;
  ActionPhase action_phase = ActionPhase::None;
  SelectionRole selection_role = SelectionRole::None;
  bool finish = false;
  bool cancel = false;
  int effect = -1;
  int turn = 0;
  int phase = 0;
  int selection_stage = 0;
  int selected_count = 0;
  uint8_t choice = 0;
  uint8_t payload_size = 0;
  uint8_t payload[5] = {};
  std::string source_spec;
  PlayerId source_spec_player = 0;
  std::string candidate_spec;
  PlayerId candidate_spec_player = 0;
};

class SpecInfo {
public:
  uint16_t index;
  CardId cid;
};

class Card {
  friend class YGOProEnvImpl;

protected:
  CardCode code_ = 0;
  uint32_t alias_;
  uint64_t setcode_;
  uint32_t type_;
  uint32_t level_;
  uint32_t lscale_;
  uint32_t rscale_;
  int32_t attack_;
  int32_t defense_;
  uint32_t race_;
  uint32_t attribute_;
  uint32_t link_marker_;
  // uint32_t category_;
  std::string name_;
  std::string desc_;
  std::vector<std::string> strings_;

  uint32_t data_ = 0;

  uint32_t status_ = 0;
  PlayerId controler_ = 0;
  uint32_t location_ = 0;
  uint32_t sequence_ = 0;
  uint32_t position_ = 0;
  uint32_t counter_ = 0;

public:
  Card() = default;

  Card(CardCode code, uint32_t alias, uint64_t setcode, uint32_t type,
       uint32_t level, uint32_t lscale, uint32_t rscale, int32_t attack,
       int32_t defense, uint32_t race, uint32_t attribute, uint32_t link_marker,
       const std::string &name, const std::string &desc,
       const std::vector<std::string> &strings)
      : code_(code), alias_(alias), setcode_(setcode), type_(type),
        level_(level), lscale_(lscale), rscale_(rscale), attack_(attack),
        defense_(defense), race_(race), attribute_(attribute),
        link_marker_(link_marker), name_(name), desc_(desc), strings_(strings) {
  }

  ~Card() = default;

  void set_location(uint32_t location) {
    controler_ = location & 0xff;
    location_ = (location >> 8) & 0xff;
    sequence_ = (location >> 16) & 0xff;
    position_ = (location >> 24) & 0xff;
  }

  const std::string &name() const { return name_; }
  const std::string &desc() const { return desc_; }
  const uint32_t &type() const { return type_; }
  const uint32_t &level() const { return level_; }
  const std::vector<std::string> &strings() const { return strings_; }

  std::string get_spec(bool opponent) const {
    return ls_to_spec(location_, sequence_, position_, opponent);
  }

  std::string get_spec(PlayerId player) const {
    return get_spec(player != controler_);
  }

  std::string get_position() const { return position_to_string(position_); }

  std::string get_effect_description(CardCode code, int effect_idx) const {
    if (code == 0) {
      return get_system_string(effect_idx);
    }
    if (effect_idx == 0) {
      return "default";
    }
    effect_idx -= CARD_EFFECT_OFFSET;
    if (effect_idx < 0) {
      throw std::runtime_error(
          fmt::format("Invalid effect index: {}", effect_idx));
    }
    auto s = strings_[effect_idx];
    if (s.empty()) {
      return "effect " + std::to_string(effect_idx);
    }
    return s;
  }
};

struct MDuel {
  intptr_t pduel;
  uint64_t seed;
  std::vector<CardCode> main_deck0;
  std::vector<CardCode> extra_deck0;
  std::string deck_name0;
  std::vector<CardCode> main_deck1;
  std::vector<CardCode> extra_deck1;
  std::string deck_name1;
};

inline Card db_query_card(const SQLite::Database &db, CardCode code) {
  SQLite::Statement query1(db, "SELECT * FROM datas WHERE id=?");
  query1.bind(1, code);
  bool found = query1.executeStep();
  if (!found) {
    std::string msg = "[db_query_card] Card not found: " + std::to_string(code);
    throw std::runtime_error(msg);
  }

  uint32_t alias = query1.getColumn("alias");
  uint64_t setcode = query1.getColumn("setcode").getInt64();
  uint32_t type = query1.getColumn("type");
  uint32_t level_ = query1.getColumn("level");
  uint32_t level = level_ & 0xff;
  uint32_t lscale = (level_ >> 24) & 0xff;
  uint32_t rscale = (level_ >> 16) & 0xff;
  int32_t attack = query1.getColumn("atk");
  int32_t defense = query1.getColumn("def");
  uint32_t link_marker = 0;
  if (type & TYPE_LINK) {
    link_marker = defense;
    defense = 0;
  }
  uint32_t race = query1.getColumn("race");
  uint32_t attribute = query1.getColumn("attribute");

  SQLite::Statement query2(db, "SELECT * FROM texts WHERE id=?");
  query2.bind(1, code);
  query2.executeStep();

  std::string name = query2.getColumn(1);
  std::string desc = query2.getColumn(2);
  std::vector<std::string> strings;
  for (int i = 3; i < query2.getColumnCount(); ++i) {
    std::string str = query2.getColumn(i);
    strings.push_back(str);
  }
  return Card(code, alias, setcode, type, level, lscale, rscale, attack,
              defense, race, attribute, link_marker, name, desc, strings);
}

inline card_data db_query_card_data(const SQLite::Database &db, CardCode code) {
  SQLite::Statement query(db, "SELECT * FROM datas WHERE id=?");
  query.bind(1, code);
  query.executeStep();
  card_data card;
  card.code = code;
  card.alias = query.getColumn("alias");
  uint64_t setcode = query.getColumn("setcode").getInt64();
  card.set_setcode(setcode);
  card.type = query.getColumn("type");
  uint32_t level_ = query.getColumn("level");
  card.level = level_ & 0xff;
  card.lscale = (level_ >> 24) & 0xff;
  card.rscale = (level_ >> 16) & 0xff;
  card.attack = query.getColumn("atk");
  card.defense = query.getColumn("def");
  if (card.type & TYPE_LINK) {
    card.link_marker = card.defense;
    card.defense = 0;
  } else {
    card.link_marker = 0;
  }
  card.race = query.getColumn("race");
  card.attribute = query.getColumn("attribute");
  return card;
}

struct card_script {
  byte *buf;
  int len;
};

static ankerl::unordered_dense::map<CardCode, Card> cards_;
static ankerl::unordered_dense::map<CardCode, CardId> card_ids_;
static ankerl::unordered_dense::map<CardCode, card_data> cards_data_;
static ankerl::unordered_dense::map<std::string, card_script> cards_script_;
static ankerl::unordered_dense::map<std::string, std::vector<CardCode>>
    main_decks_;
static ankerl::unordered_dense::map<std::string, std::vector<CardCode>>
    extra_decks_;
static std::vector<std::string> deck_names_;
static ankerl::unordered_dense::map<std::string, int> deck_names_ids_;
static std::vector<std::vector<std::string>> deck_clusters_;

inline std::vector<CardCode> announce_card_candidates(
    const std::vector<uint32_t> &opcodes, size_t limit) {
  std::vector<std::pair<CardId, CardCode>> ranked;
  ranked.reserve(cards_data_.size());
  for (const auto &[code, card] : cards_data_) {
    const auto id_it = card_ids_.find(code);
    if (id_it == card_ids_.end() ||
        !evaluate_announce_card_filter(card, opcodes)) {
      continue;
    }
    ranked.push_back({id_it->second, code});
  }
  std::sort(ranked.begin(), ranked.end());

  std::vector<CardCode> candidates;
  std::unordered_set<CardCode> seen;
  candidates.reserve(std::min(limit, ranked.size()));
  const auto append = [&](CardCode code) {
    if (candidates.size() >= limit || !seen.insert(code).second) {
      return;
    }
    const auto card_it = cards_data_.find(code);
    if (card_it != cards_data_.end() &&
        evaluate_announce_card_filter(card_it->second, opcodes)) {
      candidates.push_back(code);
    }
  };

  for (size_t i = 1; i < opcodes.size(); ++i) {
    if (opcodes[i] == OPCODE_ISCODE) {
      append(opcodes[i - 1]);
    }
  }
  for (const auto &[id, code] : ranked) {
    append(code);
  }
  if (candidates.empty()) {
    throw std::runtime_error("No declarable cards match announce-card filter");
  }
  return candidates;
}

inline const Card &c_get_card(CardCode code) {
  auto it = cards_.find(code);
  if (it != cards_.end()) {
    return it->second;
  }
  throw std::runtime_error("[c_get_card] Card not found: " + std::to_string(code));
}

inline CardId &c_get_card_id_impl(
    CardCode code, const char *function, int line) {
  auto it = card_ids_.find(code);
  if (it != card_ids_.end()) {
    return it->second;
  }
  throw std::runtime_error(fmt::format(
      "[c_get_card_id] Card not found: {} at {}:{}", code, function, line));
}

#define c_get_card_id(code) c_get_card_id_impl((code), __func__, __LINE__)

inline void sort_extra_deck(std::vector<CardCode> &deck) {
  std::vector<CardCode> c;
  std::vector<std::pair<CardCode, int>> fusion, xyz, synchro, link;

  for (auto code : deck) {
    const Card &cc = c_get_card(code);
    if (cc.type() & TYPE_FUSION) {
      fusion.push_back({code, cc.level()});
    } else if (cc.type() & TYPE_XYZ) {
      xyz.push_back({code, cc.level()});
    } else if (cc.type() & TYPE_SYNCHRO) {
      synchro.push_back({code, cc.level()});
    } else if (cc.type() & TYPE_LINK) {
      link.push_back({code, cc.level()});
    } else {
      throw std::runtime_error("Not extra deck card");
    }
  }

  auto cmp = [](const std::pair<CardCode, int> &a,
                const std::pair<CardCode, int> &b) {
    return a.second < b.second;
  };
  std::sort(fusion.begin(), fusion.end(), cmp);
  std::sort(xyz.begin(), xyz.end(), cmp);
  std::sort(synchro.begin(), synchro.end(), cmp);
  std::sort(link.begin(), link.end(), cmp);

  for (const auto &tc : fusion) {
    c.push_back(tc.first);
  }
  for (const auto &tc : xyz) {
    c.push_back(tc.first);
  }
  for (const auto &tc : synchro) {
    c.push_back(tc.first);
  }
  for (const auto &tc : link) {
    c.push_back(tc.first);
  }

  deck = c;
}

inline void preload_deck(const SQLite::Database &db,
                         const std::vector<CardCode> &deck) {
  for (const auto &code : deck) {
    auto it = cards_.find(code);
    if (it == cards_.end()) {
      cards_[code] = db_query_card(db, code);
      if (card_ids_.find(code) == card_ids_.end()) {
        throw std::runtime_error("Card not found in code list: " +
                                 std::to_string(code));
      }
    }

    auto it2 = cards_data_.find(code);
    if (it2 == cards_data_.end()) {
      cards_data_[code] = db_query_card_data(db, code);
    }
  }
}

inline uint32 card_reader_callback(CardCode code, card_data *card) {
  auto it = cards_data_.find(code);
  if (it == cards_data_.end()) {
    fmt::println("[card_reader_callback] Card not found: " + std::to_string(code));
    throw std::runtime_error("[card_reader_callback] Card not found: " + std::to_string(code));
  }
  *card = it->second;
  return 0;
}

inline byte *read_card_script(const std::string &path, int *lenptr) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    *lenptr = 0;
    return nullptr;
  }
  file.seekg(0, std::ios::end);
  int len = file.tellg();
  file.seekg(0, std::ios::beg);
  byte *buf = new byte[len];
  file.read((char *)buf, len);
  *lenptr = len;
  return buf;
}

inline byte *script_reader_callback(const char *name, int *lenptr) {
  std::string path(name);
  auto it = cards_script_.find(path);
  if (it == cards_script_.end()) {
    fmt::println("[script_reader_callback] Script not found: " + path);
    throw std::runtime_error("[script_reader_callback] Script not found: " + path);
  }
  *lenptr = it->second.len;
  return it->second.buf;
}

inline std::string deck_cluster_key(const std::string &name) {
  constexpr std::string_view prefix = "cluster_";
  if (name.rfind(prefix.data(), 0) != 0) {
    return name;
  }
  const auto separator = name.find("__", prefix.size());
  if (separator == std::string::npos || separator == prefix.size()) {
    return name;
  }
  return name.substr(0, separator);
}

static void init_module(const std::string &db_path,
                        const std::string &code_list_file,
                        const std::map<std::string, std::string> &decks) {
  main_decks_.clear();
  extra_decks_.clear();
  deck_names_.clear();
  deck_names_ids_.clear();
  deck_clusters_.clear();

  // parse code from code_list_file
  SQLite::Database db(db_path, SQLite::OPEN_READONLY);

  auto start = std::chrono::steady_clock::now();

  std::ifstream file(code_list_file);
  std::string line;
  int i = 0;
  CardCode code;
  int has_script, script_len;
  while (std::getline(file, line)) {
    i++;
    std::istringstream iss(line);
    if (!(iss >> code >> has_script)) {
        std::cerr << "Failed to parse line in code_list: " << line << std::endl;
        continue;
    }
    card_ids_[code] = i;
    cards_[code] = db_query_card(db, code);
    cards_data_[code] = db_query_card_data(db, code);
    if (has_script) {
      std::string path = "./script/c" + std::to_string(code) + ".lua";
      byte *buf = read_card_script(path, &script_len);
      cards_script_[path] = {buf, script_len};
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto milliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  // fmt::println("load {} cards in {}ms", cards_data_.size(), milliseconds);

  std::map<std::string, std::vector<std::string>> clusters;
  for (const auto &[name, deck] : decks) {
    auto [main_deck, extra_deck, side_deck] = read_decks(deck);
    main_decks_[name] = main_deck;
    extra_decks_[name] = extra_deck;
    if (name[0] != '_') {
      deck_names_.push_back(name);
      deck_names_ids_[name] = deck_names_.size() - 1;
      clusters[deck_cluster_key(name)].push_back(name);
    }
  }
  deck_clusters_.reserve(clusters.size());
  for (auto &[cluster, names] : clusters) {
    deck_clusters_.push_back(std::move(names));
  }

  for (auto &[name, deck] : extra_decks_) {
    sort_extra_deck(deck);
  }

  card_data card;
  cards_data_[0] = card;

  std::vector<std::string> preload = {
    "./script/constant.lua",
    "./script/utility.lua",
    "./script/procedure.lua",
  };
  for (const auto &path : preload) {
    byte *buf = read_card_script(path, &script_len);
    cards_script_[path] = {buf, script_len};
  }
  cards_script_["./script/c0.lua"] = {nullptr, 0};

  set_card_reader(card_reader_callback);
  set_script_reader(script_reader_callback);
}

inline std::string getline() {
  char *line = nullptr;
  size_t len = 0;
  ssize_t read;

  read = getline(&line, &len, stdin);

  if (read != -1) {
    // Remove line ending character(s)
    if (line[read - 1] == '\n')
      line[read - 1] = '\0'; // Replace newline character with null terminator
    else if (line[read - 2] == '\r' && line[read - 1] == '\n') {
      line[read - 2] = '\0'; // Replace carriage return and newline characters
                             // with null terminator
      line[read - 1] = '\0';
    }

    std::string input(line);
    free(line);
    return input;
  } else {
    exit(0);
  }

  free(line);
  return "";
}

class Player {
  friend class YGOProEnvImpl;

protected:
  const std::string nickname_;
  const int init_lp_;
  const PlayerId duel_player_;
  const bool verbose_;

  bool seen_waiting_ = false;

public:
  Player(const std::string &nickname, int init_lp, PlayerId duel_player,
         bool verbose = false)
      : nickname_(nickname), init_lp_(init_lp), duel_player_(duel_player),
        verbose_(verbose) {}
  virtual ~Player() = default;

  void notify(const std::string &text) {
    if (verbose_) {
      fmt::println("{} {}", duel_player_, text);
    }
  }

  const int &init_lp() const { return init_lp_; }

  virtual int think(const std::vector<LegalAction> &actions) = 0;
};

class GreedyAI : public Player {
protected:
public:
  GreedyAI(const std::string &nickname, int init_lp, PlayerId duel_player,
           bool verbose = false)
      : Player(nickname, init_lp, duel_player, verbose) {}

  int think(const std::vector<LegalAction> &actions) override { return 0; }
};

class RandomAI : public Player {
protected:
  std::mt19937 gen_;
  std::uniform_int_distribution<int> dist_;

public:
  RandomAI(int max_options, int seed, const std::string &nickname, int init_lp,
           PlayerId duel_player, bool verbose = false)
      : Player(nickname, init_lp, duel_player, verbose), gen_(seed),
        dist_(0, max_options - 1) {}

  int think(const std::vector<LegalAction> &actions) override {
    return dist_(gen_) % actions.size();
  }
};

class HumanPlayer : public Player {
protected:
public:
  HumanPlayer(const std::string &nickname, int init_lp, PlayerId duel_player,
              bool verbose = false)
      : Player(nickname, init_lp, duel_player, verbose) {}

  int think(const std::vector<LegalAction> &actions) override {
    while (true) {
      std::string input = getline();
      if (input == "quit") {
        exit(0);
      }
      int idx = -1;
      try {
        idx = std::stoi(input) - 1;
      } catch (std::invalid_argument &e) {
        fmt::println("{} Invalid input: {}", duel_player_, input);
        continue;
      }
      if (idx >= 0 && idx < actions.size()) {
        return idx;
      } else {
        fmt::println("{} Choose from {} actions", duel_player_, actions.size());
      }
    }
  }
};

class YGOProEnvFns {
public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("deck1"_.Bind(std::string("OldSchool")),
                    "deck2"_.Bind(std::string("OldSchool")), "player"_.Bind(-1),
                    "deck_schedule"_.Bind(std::string("independent")),
                    "anchor_deck"_.Bind(std::string("")),
                    "anchor_probability_percent"_.Bind(50),
                    "play_mode"_.Bind(std::string("bot")),
                    "verbose"_.Bind(false), "max_options"_.Bind(64),
                    "max_cards"_.Bind(80), "n_history_actions"_.Bind(16),
                    "record"_.Bind(false), "async_reset"_.Bind(false),
                    "greedy_reward"_.Bind(false), "timeout"_.Bind(600),
                    "oppo_info"_.Bind(false), "max_steps"_.Bind(1000));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config &conf) {
    int n_action_feats = 12;
    return MakeDict(
        "obs:cards_"_.Bind(Spec<uint8_t>({conf["max_cards"_] * 2, 41})),
        "obs:global_"_.Bind(Spec<uint8_t>({23})),
        "obs:actions_"_.Bind(
            Spec<uint8_t>({conf["max_options"_], n_action_feats})),
        "obs:h_actions_"_.Bind(
            Spec<uint8_t>({conf["n_history_actions"_], n_action_feats + 2})),
        "obs:action_ir_"_.Bind(
            Spec<uint8_t>({conf["max_options"_], kActionIRFeatures})),
        "obs:action_single_refs_"_.Bind(
            Spec<uint8_t>({conf["max_options"_], kActionSingleRefs})),
        "obs:action_group_refs_"_.Bind(
            Spec<uint8_t>({conf["max_options"_], kActionGroupRoles,
                           kMaxActionRoleMembers})),
        "obs:action_group_mask_"_.Bind(
            Spec<uint8_t>({conf["max_options"_], kActionGroupRoles,
                           kMaxActionRoleMembers})),
        "obs:selection_"_.Bind(Spec<uint8_t>({kSelectionFeatures})),
        "obs:public_events_"_.Bind(
            Spec<uint8_t>({conf["n_history_actions"_], kPublicEventFeatures})),
        "obs:public_event_refs_"_.Bind(
            Spec<uint8_t>({conf["n_history_actions"_], kPublicEventRefs})),
        "obs:mask_"_.Bind(Spec<uint8_t>({conf["max_cards"_] * 2, 14})),
        "info:num_options"_.Bind(Spec<int>({}, {0, conf["max_options"_]})),
        "info:to_play"_.Bind(Spec<int>({}, {0, 1})),
        "info:is_selfplay"_.Bind(Spec<int>({}, {0, 1})),
        "info:win_reason"_.Bind(Spec<int>({}, {-1, 1})),
        "info:step_time"_.Bind(Spec<double>({2})),
        "info:deck"_.Bind(Spec<int>({2})),
        "info:anchor_seat"_.Bind(Spec<int>({}, {0, 2}))
      );
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config &conf) {
    return MakeDict(
        "action"_.Bind(Spec<int>({}, {0, conf["max_options"_] - 1})));
  }
};

using YGOProEnvSpec = EnvSpec<YGOProEnvFns>;

enum PlayMode { kHuman, kSelfPlay, kRandomBot, kGreedyBot, kCount };

// parse play modes seperated by '+'
inline std::vector<PlayMode> parse_play_modes(const std::string &play_mode) {
  std::vector<PlayMode> modes;
  std::istringstream ss(play_mode);
  std::string token;
  while (std::getline(ss, token, '+')) {
    if (token == "human") {
      modes.push_back(kHuman);
    } else if (token == "self") {
      modes.push_back(kSelfPlay);
    } else if (token == "bot") {
      modes.push_back(kGreedyBot);
    } else if (token == "random") {
      modes.push_back(kRandomBot);
    } else {
      throw std::runtime_error("Unknown play mode: " + token);
    }
  }
  // human mode can't be combined with other modes
  if (std::find(modes.begin(), modes.end(), kHuman) != modes.end() &&
      modes.size() > 1) {
    throw std::runtime_error("Human mode can't be combined with other modes");
  }
  return modes;
}

// rules = 1, Traditional
// rules = 0, Default
// rules = 4, Link
// rules = 5, MR5
constexpr int32_t rules_ = 5;
constexpr int32_t duel_options_ = ((rules_ & 0xFF) << 16) + (0 & 0xFFFF);


class YGOProEnvImpl {
protected:
  const EnvSpec<YGOProEnvFns> spec_;

  constexpr static int init_lp_ = 8000;
  constexpr static int startcount_ = 5;
  constexpr static int drawcount_ = 1;

  const std::string deck1_;
  const std::string deck2_;
  const std::string deck_schedule_;
  const std::string anchor_deck_;
  const int anchor_probability_percent_;

  std::vector<uint32> main_deck0_;
  std::vector<uint32> main_deck1_;
  std::vector<uint32> extra_deck0_;
  std::vector<uint32> extra_deck1_;

  std::string deck_name_[2] = {"", ""};
  std::string nickname_[2] = {"Alice", "Bob"};

  const std::vector<PlayMode> play_modes_;

  // if play_mode_ == 'bot' or 'human', player_ is the order of the ai player
  // -1 means random, 0 and 1 means the first and second player respectively
  const int player_;

  PlayMode play_mode_;
  bool verbose_ = false;

  PlayerId ai_player_;

  intptr_t pduel_ = 0;
  std::unique_ptr<Player> players_[2]; //  abstract class must be pointer

  std::uniform_int_distribution<uint64_t> dist_int_;
  bool done_{true};
  long step_count_{0};
  bool duel_started_{false};
  uint32_t eng_flag_{0};
  uint32_t disabled_field_{0};

  PlayerId winner_;
  uint8_t win_reason_;
  const bool greedy_reward_;

  int lp_[2];

  // turn player
  PlayerId tp_;
  int current_phase_;
  int turn_count_;

  int msg_;
  std::vector<LegalAction> legal_actions_;
  PlayerId to_play_;
  std::function<void(int)> callback_;

  byte data_[4096];
  int dp_ = 0;
  int dl_ = 0;

  byte query_buf_[4096];
  int qdp_ = 0;

  byte resp_buf_[128];

  using IdleCardSpec = std::tuple<CardCode, std::string, uint32_t>;

  // chain
  PlayerId chaining_player_;

  const int n_history_actions_;

  // circular buffer for history actions
  TArray<uint8_t> history_actions_1_;
  TArray<uint8_t> history_actions_2_;
  int ha_p_1_ = 0;
  int ha_p_2_ = 0;

  std::unordered_set<std::string> revealed_;
  std::vector<PublicEvent> public_events_;

  std::string pending_source_spec_;
  PlayerId pending_source_spec_player_ = 0;
  CardId pending_source_cid_ = 0;
  int pending_source_effect_ = -1;

  // multi select
  int ms_idx_ = -1;
  int ms_mode_ = 0;
  int ms_min_ = 0;
  int ms_max_ = 0;
  int ms_must_ = 0;
  std::vector<std::string> ms_specs_;
  std::vector<std::vector<int>> ms_combs_;
  ankerl::unordered_dense::map<std::string, int> ms_spec2idx_;
  std::vector<int> ms_r_idxs_;

  // discard hand cards
  bool discard_hand_ = false;

  // replay
  bool record_ = false;
  FILE* fp_ = nullptr;
  bool is_recording = false;

  // MSG_SELECT_COUNTER
  int n_counters_ = 0;

  std::mt19937 gen_;

  std::mt19937 duel_gen_;


public:
  // step return
  float ret_reward_ = 0;
  int ret_win_reason_ = 0;

  YGOProEnvImpl();

  YGOProEnvImpl(const EnvSpec<YGOProEnvFns> &spec, uint64_t env_seed)
      : spec_(spec), dist_int_(0, 0xffffffff),
        deck1_(spec.config["deck1"_]), deck2_(spec.config["deck2"_]),
        deck_schedule_(spec.config["deck_schedule"_]),
        anchor_deck_(spec.config["anchor_deck"_]),
        anchor_probability_percent_(
            spec.config["anchor_probability_percent"_]),
        player_(spec.config["player"_]), players_{nullptr, nullptr},
        play_modes_(parse_play_modes(spec.config["play_mode"_])),
        verbose_(spec.config["verbose"_]), record_(spec.config["record"_]),
        n_history_actions_(spec.config["n_history_actions"_]),
        greedy_reward_(spec.config["greedy_reward"_]) {
    if (anchor_probability_percent_ < 0 ||
        anchor_probability_percent_ > 100) {
      throw std::runtime_error(
          "anchor_probability_percent must be between 0 and 100");
    }
    if (record_) {
      if (!verbose_) {
        throw std::runtime_error("record mode must be used with verbose mode and num_envs=1");
      }
    }
    // fmt::println("env_id: {}, seed: {}, x: {}", env_id_, seed_, dist_int_(gen_));

    gen_ = std::mt19937(env_seed);
    duel_gen_ = std::mt19937(dist_int_(gen_));

    int max_options = spec.config["max_options"_];
    int n_action_feats = spec.state_spec["obs:actions_"_].shape[1];
    history_actions_1_ = TArray<uint8_t>(Array(
        ShapeSpec(sizeof(uint8_t), {n_history_actions_, n_action_feats + 2})));
    history_actions_2_ = TArray<uint8_t>(Array(
        ShapeSpec(sizeof(uint8_t), {n_history_actions_, n_action_feats + 2})));
  }

  int max_options() const { return spec_.config["max_options"_]; }

  int max_cards() const { return spec_.config["max_cards"_]; }

  bool done() const { return done_; }

  bool random_mode() const { return play_modes_.size() > 1; }

  bool self_play() const {
    return std::find(play_modes_.begin(), play_modes_.end(), kSelfPlay) !=
           play_modes_.end();
  }

  void update_time_stat(const clock_t& start, uint64_t time_count, double& time_stat) {
    double seconds = static_cast<double>(clock() - start) / CLOCKS_PER_SEC;
    time_stat = time_stat * (static_cast<double>(time_count) /
      (time_count + 1)) + seconds / (time_count + 1);
  }

  // void update_time_stat(const std::string& deck, double seconds) {
  //   uint64_t& time_count = deck_time_count_[deck];
  //   double& time_stat = deck_time_[deck];
  //   time_stat = time_stat * (static_cast<double>(time_count) /
  //     (time_count + 1)) + seconds / (time_count + 1);
  //   time_count++;
  // }

  std::string sample_deck_name(
      const std::string &configured_name, std::mt19937 &gen) const {
    if (configured_name != "random") {
      return configured_name;
    }
    if (deck_names_.empty()) {
      throw std::runtime_error("No decks are available for random sampling");
    }
    std::uniform_int_distribution<size_t> distribution(
        0, deck_names_.size() - 1);
    return deck_names_[distribution(gen)];
  }

  std::string sample_cluster_deck_name(
      const std::string &configured_name, std::mt19937 &gen) const {
    if (configured_name != "random") {
      return configured_name;
    }
    if (deck_clusters_.empty()) {
      throw std::runtime_error(
          "No deck clusters are available for random sampling");
    }
    std::uniform_int_distribution<size_t> cluster_distribution(
        0, deck_clusters_.size() - 1);
    const auto &cluster = deck_clusters_[cluster_distribution(gen)];
    std::uniform_int_distribution<size_t> variant_distribution(
        0, cluster.size() - 1);
    return cluster[variant_distribution(gen)];
  }

  std::pair<std::string, std::string> sample_deck_pair(
      std::mt19937 &gen) const {
    if (deck_schedule_ == "independent") {
      return {
          sample_deck_name(deck1_, gen),
          sample_deck_name(deck2_, gen),
      };
    }
    if (deck_schedule_ == "cluster_uniform") {
      return {
          sample_cluster_deck_name(deck1_, gen),
          sample_cluster_deck_name(deck2_, gen),
      };
    }
    if (deck_schedule_ != "anchor_half") {
      throw std::runtime_error(
          "Unknown deck schedule: " + deck_schedule_);
    }
    if (anchor_deck_.empty() ||
        main_decks_.find(anchor_deck_) == main_decks_.end()) {
      throw std::runtime_error(
          "anchor_half requires a valid anchor_deck");
    }

    std::vector<std::string> alternatives;
    alternatives.reserve(deck_names_.size());
    for (const auto &name : deck_names_) {
      if (name != anchor_deck_) {
        alternatives.push_back(name);
      }
    }
    if (alternatives.empty()) {
      throw std::runtime_error(
          "anchor_half requires at least one non-anchor deck");
    }

    std::uniform_int_distribution<int> percent(0, 99);
    std::uniform_int_distribution<size_t> pick(0, alternatives.size() - 1);
    if (percent(gen) < anchor_probability_percent_) {
      std::pair<std::string, std::string> pair = {
          anchor_deck_, alternatives[pick(gen)]};
      std::uniform_int_distribution<int> seat(0, 1);
      if (seat(gen) != 0) {
        std::swap(pair.first, pair.second);
      }
      return pair;
    }

    const auto first_index = pick(gen);
    auto second_index = first_index;
    if (alternatives.size() > 1) {
      while (second_index == first_index) {
        second_index = pick(gen);
      }
    }
    return {alternatives[first_index], alternatives[second_index]};
  }

  MDuel new_duel(uint32_t seed) {
    auto pduel = YGO_CreateDuel(seed);
    MDuel mduel{pduel, seed};
    const auto [deck_name0, deck_name1] = sample_deck_pair(duel_gen_);

    for (PlayerId i = 0; i < 2; i++) {
      YGO_SetPlayerInfo(pduel, i, init_lp_, startcount_, drawcount_);
      const auto &deck_name = i == 0 ? deck_name0 : deck_name1;
      auto [main_deck, extra_deck, loaded_deck_name] =
          load_deck(pduel, i, deck_name, duel_gen_);
      if (i == 0) {
        mduel.main_deck0 = main_deck;
        mduel.extra_deck0 = extra_deck;
        mduel.deck_name0 = loaded_deck_name;
      } else {
        mduel.main_deck1 = main_deck;
        mduel.extra_deck1 = extra_deck;
        mduel.deck_name1 = loaded_deck_name;
      }
    }
    YGO_StartDuel(pduel, duel_options_);
    return mduel;
  }

  void reset() {
    // clock_t start = clock();

    if (random_mode()) {
      play_mode_ = play_modes_[dist_int_(gen_) % play_modes_.size()];
    } else {
      play_mode_ = play_modes_[0];
    }

    if (play_mode_ != kSelfPlay) {
      if (player_ == -1) {
        ai_player_ = dist_int_(gen_) % 2;
      } else {
        ai_player_ = player_;
      }
    }

    turn_count_ = 0;
    disabled_field_ = 0;
    ms_idx_ = -1;

    history_actions_1_.Zero();
    history_actions_2_.Zero();
    ha_p_1_ = 0;
    ha_p_2_ = 0;
    public_events_.clear();
    pending_source_spec_.clear();
    pending_source_spec_player_ = 0;
    pending_source_cid_ = 0;
    pending_source_effect_ = -1;

    // clock_t _start = clock();

    intptr_t old_duel = pduel_;
    if (duel_started_) {
      YGO_EndDuel(pduel_);
    }
    MDuel mduel;
    mduel = new_duel(dist_int_(gen_));

    auto duel_seed = mduel.seed;
    pduel_ = mduel.pduel;

    deck_name_[0] = mduel.deck_name0;
    deck_name_[1] = mduel.deck_name1;
    main_deck0_ = mduel.main_deck0;
    extra_deck0_ = mduel.extra_deck0;
    main_deck1_ = mduel.main_deck1;
    extra_deck1_ = mduel.extra_deck1;

    for (PlayerId i = 0; i < 2; i++) {
      std::string nickname = i == 0 ? "Alice" : "Bob";
      if (i == ai_player_) {
        nickname = "Agent";
      }
      nickname_[i] = nickname;
      if ((play_mode_ == kHuman) && (i != ai_player_)) {
        players_[i] = std::make_unique<HumanPlayer>(nickname, init_lp_, i, verbose_);
      } else if (play_mode_ == kRandomBot) {
        players_[i] = std::make_unique<RandomAI>(max_options(), dist_int_(gen_), nickname, init_lp_, i, verbose_);
      } else {
        players_[i] = std::make_unique<GreedyAI>(nickname, init_lp_, i, verbose_);
      }
      lp_[i] = players_[i]->init_lp_;
    }

    if (record_) {
      if (is_recording && fp_ != nullptr) {
        fclose(fp_);
      }
      auto time_str = time_now();
      // Use last 4 digits of seed as unique id
      auto seed_ = duel_seed % 10000;
      std::string fname;
      while (true) {
        fname = fmt::format("./replay/a{} {:04d}.yrp", time_str, seed_);
        // check existence
        if (std::filesystem::exists(fname)) {
          seed_ = (seed_ + 1) % 10000;
        } else {
          break;
        } 
      }
      fp_ = fopen(fname.c_str(), "wb");
      if (!fp_) {
        throw std::runtime_error("Failed to open file for replay: " + fname);
      }

      is_recording = true;

      ReplayHeader rh;
      rh.id = 0x31707279;
      rh.version = 0x00001360;
      rh.flag = REPLAY_UNIFORM;
      rh.seed = duel_seed;
      rh.start_time = (unsigned int)time(nullptr);
      fwrite(&rh, sizeof(rh), 1, fp_);

      for (PlayerId i = 0; i < 2; i++) {
        uint16_t name[20];
        memset(name, 0, 40);
        std::string name_str = fmt::format("{} {}", nickname_[i], deck_name_[i]);
        if (name_str.size() > 20) {
          // truncate
          name_str = name_str.substr(0, 20);
        }
        fmt::println("name: {}", name_str);
        str_to_uint16(name_str.c_str(), name);
        fwrite(name, 40, 1, fp_);
      }

      ReplayWriteInt32(init_lp_);
      ReplayWriteInt32(startcount_);
      ReplayWriteInt32(drawcount_);
      ReplayWriteInt32(duel_options_);

      for (PlayerId i = 0; i < 2; i++) {
        auto &main_deck = i == 0 ? main_deck0_ : main_deck1_;
        auto &extra_deck = i == 0 ? extra_deck0_ : extra_deck1_;
        ReplayWriteInt32(main_deck.size());
        for (auto code : main_deck) {
          ReplayWriteInt32(code);
        }
        ReplayWriteInt32(extra_deck.size());
        for (int j = int(extra_deck.size()) - 1; j >= 0; --j) {
          ReplayWriteInt32(extra_deck[j]);
        }
      }

    }

    duel_started_ = true;
    eng_flag_ = 0;
    winner_ = 255;
    win_reason_ = 255;
    discard_hand_ = false;

    done_ = false;
    step_count_ = 0;

    // update_time_stat(_start, reset_time_count_, reset_time_2_);
    // _start = clock();

    next();

    ret_reward_ = 0;
    ret_win_reason_ = 0;
  }

  void init_multi_select(
    int min, int max, int must, const std::vector<std::string> &specs,
    int mode = 0, const std::vector<std::vector<int>> &combs = {}) {
    ms_idx_ = 0;
    ms_mode_ = mode;
    ms_min_ = min;
    ms_max_ = max;
    ms_must_ = must;
    ms_specs_ = specs;
    ms_r_idxs_.clear();
    ms_spec2idx_.clear();

    for (int j = 0; j < ms_specs_.size(); ++j) {
      const auto &spec = ms_specs_[j];
      ms_spec2idx_[spec] = j;
    }

    if (ms_mode_ == 0) {
      for (int j = 0; j < ms_specs_.size(); ++j) {
        const auto &spec = ms_specs_[j];
        legal_actions_.push_back(LegalAction::from_spec(spec));
      }
    } else {
      ms_combs_ = combs;
      _callback_multi_select_2_prepare();
    }
  }

  void handle_multi_select() {
    legal_actions_.clear();
    if (ms_mode_ == 0) {
      for (int j = 0; j < ms_specs_.size(); ++j) {
        if (ms_spec2idx_.find(ms_specs_[j]) != ms_spec2idx_.end()) {
          legal_actions_.push_back(
            LegalAction::from_spec(ms_specs_[j]));
        }
      }
      if (ms_idx_ == ms_max_ - 1) {
        if (ms_idx_ >= ms_min_) {
          legal_actions_.push_back(LegalAction::finish());
        }
        callback_ = [this](int idx) {
          _callback_multi_select(idx, true);
        };
      } else if (ms_idx_ >= ms_min_) {
        legal_actions_.push_back(LegalAction::finish());
        callback_ = [this](int idx) {
          _callback_multi_select(idx, false);
        };
      } else {
        callback_ = [this](int idx) {
          _callback_multi_select(idx, false);
        };    
      }
    } else {
      _callback_multi_select_2_prepare();
      callback_ = [this](int idx) {
        _callback_multi_select_2(idx);
      };
    }
  }

  int get_ms_spec_idx(const std::string &spec) const {
    auto it = ms_spec2idx_.find(spec);
    if (it != ms_spec2idx_.end()) {
      return it->second;
    }
    // TODO(2): find the root cause
    // print ms_spec2idx
    show_deck(0);
    show_deck(1);
    show_buffer();
    show_turn();
    fmt::println("MS: idx: {}, mode: {}, min: {}, max: {}, must: {}, specs: {}, combs: {}, r_idx: {}", ms_idx_, ms_mode_, ms_min_, ms_max_, ms_must_, ms_specs_, ms_combs_, ms_r_idxs_);
    fmt::print("ms_spec2idx: ");
    for (const auto &[k, v] : ms_spec2idx_) {
      fmt::print("({}, {}), ", k, v);
    }
    fmt::print("\n");
    return -1;
    // throw std::runtime_error("Spec not found: " + spec);
  }

  void _callback_multi_select_2(int idx) {
    const auto &action = legal_actions_[idx];
    if (action.finish_) {
      _callback_multi_select_2_finish();
      return;
    }

    const int selected_idx = get_ms_spec_idx(action.spec_);
    if (selected_idx == -1) {
      // TODO(2): find the root cause
      std::vector<std::string> specs;
      for (const auto &la : legal_actions_) {
        specs.push_back(la.spec_);
      }
      fmt::println(
          "specs: {}, idx: {}, spec: {}", specs, selected_idx, action.spec_);
      throw std::runtime_error("Spec not found");
    }

    bool can_finish = false;
    std::vector<std::vector<int>> remaining_combinations;
    for (const auto &combination : ms_combs_) {
      if (combination.empty() || combination.front() != selected_idx) {
        continue;
      }
      if (combination.size() == 1) {
        can_finish = true;
      } else {
        remaining_combinations.emplace_back(
            combination.begin() + 1, combination.end());
      }
    }

    if (can_finish) {
      remaining_combinations.push_back({});
    }
    if (remaining_combinations.empty()) {
      throw std::runtime_error("No valid continuation for multi select");
    }

    ms_r_idxs_.push_back(selected_idx);
    ms_idx_++;
    const bool has_continuation = std::any_of(
        remaining_combinations.begin(), remaining_combinations.end(),
        [](const std::vector<int> &combination) {
          return !combination.empty();
        });
    ms_combs_ = std::move(remaining_combinations);
    if (can_finish && !has_continuation) {
      _callback_multi_select_2_finish();
    }
  }

  void _callback_multi_select_2_prepare() {
    bool can_finish = false;
    std::set<int> candidates;
    for (const auto &combination : ms_combs_) {
      if (combination.empty()) {
        can_finish = true;
      } else {
        candidates.insert(combination.front());
      }
    }
    for (const auto index : candidates) {
      const auto &spec = ms_specs_[index];
      legal_actions_.push_back(LegalAction::from_spec(spec));
    }
    if (can_finish) {
      legal_actions_.push_back(LegalAction::finish());
    }
  }

  void _callback_multi_select_2_finish() {
    ms_idx_ = -1;
    resp_buf_[0] = ms_r_idxs_.size() + ms_must_;
    for (int i = 0; i < ms_must_; ++i) {
      resp_buf_[i + 1] = 0;
    }
    for (int i = 0; i < ms_r_idxs_.size(); ++i) {
      resp_buf_[i + ms_must_ + 1] = ms_r_idxs_[i];
    }
    YGO_SetResponseb(pduel_, resp_buf_);
  }

  void _callback_multi_select(int idx, bool finish) {
    const auto &action = legal_actions_[idx];
    // fmt::println("Select card: {}, finish: {}", option, finish);
    if (action.finish_) {
      finish = true;
    } else {
      idx = get_ms_spec_idx(action.spec_);
      if (idx != -1) {
        ms_r_idxs_.push_back(idx);
      } else {
        // TODO(2): find the root cause
        std::vector<std::string> specs;
        for (const auto &la : legal_actions_) {
          specs.push_back(la.spec_);
        }
        fmt::println("specs: {}, idx: {}, spec: {}", specs, idx, action.spec_);
        ms_idx_ = -1;
        resp_buf_[0] = ms_min_;
        for (int i = 0; i < ms_min_; ++i) {
          resp_buf_[i + 1] = i;
        }
        YGO_SetResponseb(pduel_, resp_buf_);
        return;
      }
    }
    if (finish) {
      ms_idx_ = -1;
      resp_buf_[0] = ms_r_idxs_.size();
      for (int i = 0; i < ms_r_idxs_.size(); ++i) {
        resp_buf_[i + 1] = ms_r_idxs_[i];
      }
      YGO_SetResponseb(pduel_, resp_buf_);
    } else {
      ms_idx_++;
      ms_spec2idx_.erase(action.spec_);
    }
  }

  SelectionRole selection_role_for_msg(int msg) const {
    if (msg == MSG_SELECT_TRIBUTE) {
      return SelectionRole::Tribute;
    }
    if (msg == MSG_SELECT_SUM) {
      return SelectionRole::Material;
    }
    if (msg == MSG_SELECT_CARD || msg == MSG_SELECT_UNSELECT_CARD) {
      return SelectionRole::Candidate;
    }
    return SelectionRole::None;
  }

  bool is_candidate_selection_msg(int msg) const {
    return msg == MSG_SELECT_CARD || msg == MSG_SELECT_TRIBUTE ||
           msg == MSG_SELECT_SUM || msg == MSG_SELECT_UNSELECT_CARD;
  }

  bool is_source_action_msg(int msg) const {
    return msg == MSG_SELECT_IDLECMD || msg == MSG_SELECT_BATTLECMD ||
           msg == MSG_SELECT_CHAIN || msg == MSG_SELECT_EFFECTYN;
  }

  std::string spec_for_observer(
      const std::string &spec, PlayerId source_player,
      PlayerId observer) const {
    if (spec.empty()) {
      return "";
    }
    const bool points_to_opponent = spec[0] == 'o';
    const auto core = points_to_opponent ? spec.substr(1) : spec;
    const PlayerId controller =
        points_to_opponent ? 1 - source_player : source_player;
    return controller == observer ? core : "o" + core;
  }

  void push_public_event(const PublicEvent &event) {
    public_events_.insert(public_events_.begin(), event);
    if (public_events_.size() > static_cast<size_t>(n_history_actions_)) {
      public_events_.pop_back();
    }
  }

  void record_protocol_event(
      PlayerId player, int msg, const std::vector<uint8_t> &payload) {
    PublicEvent event;
    event.actor = player;
    event.msg = msg;
    event.turn = turn_count_;
    event.phase = current_phase_;
    event.payload_size = static_cast<uint8_t>(
        std::min<size_t>(payload.size(), 5));
    for (int i = 0; i < event.payload_size; ++i) {
      event.payload[i] = payload[i];
    }
    push_public_event(event);
  }

  void record_public_event(PlayerId player, const LegalAction &action) {
    if (action.act_ == ActionAct::Cancel) {
      return;
    }

    PublicEvent event;
    event.actor = player;
    event.msg = action.msg_;
    event.act = action.act_;
    event.action_phase = action.phase_;
    event.selection_role = selection_role_for_msg(action.msg_);
    event.finish = action.finish_;
    event.cancel = action.act_ == ActionAct::Cancel;
    event.effect = action.effect_;
    event.turn = turn_count_;
    event.phase = current_phase_;
    event.selection_stage = std::max(0, ms_idx_);
    event.selected_count = static_cast<int>(ms_r_idxs_.size());
    event.choice = action.number_;
    if (event.choice == 0 && action.attribute_ != 0) {
      event.choice = attribute_to_id(action.attribute_);
    }

    if (is_candidate_selection_msg(action.msg_)) {
      event.candidate_spec = action.spec_;
      event.candidate_spec_player = player;
      event.source_spec = pending_source_spec_;
      event.source_spec_player = pending_source_spec_player_;
    } else if (!action.spec_.empty()) {
      event.source_spec = action.spec_;
      event.source_spec_player = player;
    }

    push_public_event(event);

    if (is_source_action_msg(action.msg_) && !action.spec_.empty() &&
        action.act_ != ActionAct::Cancel) {
      pending_source_spec_ = action.spec_;
      pending_source_spec_player_ = player;
      pending_source_cid_ = action.cid_;
      pending_source_effect_ = action.effect_;
    } else if (action.phase_ != ActionPhase::None) {
      pending_source_spec_.clear();
      pending_source_cid_ = 0;
      pending_source_effect_ = -1;
    }
  }

  void update_history_actions(PlayerId player, const LegalAction& action) {
    if (action.act_ == ActionAct::Cancel) {
      return;
    }
    auto& ha_p = player == 0 ? ha_p_1_ : ha_p_2_;
    auto& history_actions = player == 0 ? history_actions_1_ : history_actions_2_;
    ha_p--;
    if (ha_p < 0) {
      ha_p = n_history_actions_ - 1;
    }
    history_actions[ha_p].Zero();
    _set_obs_action(history_actions, ha_p, action);
    // Spec index not available in history actions
    history_actions[ha_p](0) = 0;
    // history_actions[ha_p](12) = static_cast<uint8_t>(player);
    history_actions[ha_p](12) = static_cast<uint8_t>(turn_count_);
    history_actions[ha_p](13) = static_cast<uint8_t>(phase_to_id(current_phase_));
  }

  void show_deck(const std::vector<CardCode> &deck, const std::string &prefix) const {
    fmt::print("{} deck: [", prefix);
    for (int i = 0; i < deck.size(); i++) {
      fmt::print(" '{}'", c_get_card(deck[i]).name());
    }
    fmt::print(" ]\n");
  }

  void show_turn() const {
    fmt::println("turn: {}, phase: {}, tplayer: {}", turn_count_, phase_to_string(current_phase_), tp_);
  }

  void show_buffer() const {
    fmt::println("msg: {}, dp: {}, dl: {}", msg_to_string(msg_), dp_, dl_);
    for (int i = 0; i < dl_; ++i) {
      fmt::print("{:02x} ", data_[i]);
    }
    fmt::print("\n");
  }

  void show_deck(PlayerId player) const {
    fmt::print("Player {}'s deck: {}\n", player, deck_name_[player]);
    // show_deck(player == 0 ? main_deck0_ : main_deck1_, "Main");
    // show_deck(player == 0 ? extra_deck0_ : extra_deck1_, "Extra");
  }

  void show_history_actions(PlayerId player) const {
    const auto &ha = player == 0 ? history_actions_1_ : history_actions_2_;
    // print card ids of history actions
    for (int i = 0; i < n_history_actions_; ++i) {
      fmt::print("history {}\n", i);
      uint8_t msg_id = uint8_t(ha(i, 3));
      int msg = _msgs[msg_id - 1];
      fmt::print("msg: {},", msg_to_string(msg));
      uint8_t v1 = ha(i, 1);
      uint8_t v2 = ha(i, 2);
      CardId card_id = (static_cast<CardId>(v1) << 8) + static_cast<CardId>(v2);
      fmt::print(" {};", card_id);
      for (int j = 4; j < ha.Shape()[1]; j++) {
        fmt::print(" {}", uint8_t(ha(i, j)));
      }
      fmt::print("\n");
    }
  }

  void step(int idx) {
    const auto selected_action = legal_actions_.at(idx);
    callback_(idx);
    update_history_actions(to_play_, selected_action);
    record_public_event(to_play_, selected_action);

    PlayerId player = to_play_;

    if (verbose_) {
      show_decision(idx);
    }

    if (ms_idx_ != -1) {
      handle_multi_select();
    } else {
      next();
    }

    step_count_++;
    if (!done_ && (step_count_ >= spec_.config["max_steps"_])) {
      _duel_end(2, 0);
      done_ = true;
      legal_actions_.clear();
    }

    float reward = 0;
    int reason = 0;
    if (done_) {
      if (winner_ > 1) {
        if (record_ && is_recording && fp_ != nullptr) {
          fclose(fp_);
          is_recording = false;
        }
        ret_reward_ = 0;
        ret_win_reason_ = 0;
        return;
      }
      float base_reward;
      if (greedy_reward_) {
        if (winner_ == 0) {
          if (turn_count_ <= 1) {
            // FTK
            base_reward = 16.0;
          } else if (turn_count_ <= 3) {
            base_reward = 8.0;
          } else if (turn_count_ <= 5) {
            base_reward = 4.0;
          } else if (turn_count_ <= 7) {
            base_reward = 2.0;
          } else {
            base_reward = 0.5 + 1.0 / (turn_count_ - 7);
          }
        } else {
          if (turn_count_ <= 1) {
            base_reward = 8.0;
          } else if (turn_count_ <= 3) {
            base_reward = 4.0;
          } else if (turn_count_ <= 5) {
            base_reward = 2.0;
          } else {
            base_reward = 0.5 + 1.0 / (turn_count_ - 5);
          }
        }
      } else {
        base_reward = 1.0;
      }

      if (play_mode_ == kSelfPlay) {
        // if (spec_.config["oppo_info"_]) {
        if (false) {
          reward = winner_ == 0 ? base_reward : -base_reward;
        } else {
          // to_play_ is the previous player
          reward = winner_ == player ? base_reward : -base_reward;
        }
      } else {
        reward = winner_ == ai_player_ ? base_reward : -base_reward;
      }

      if (win_reason_ == 0x01) {
        reason = 1;
      } else if (win_reason_ == 0x02) {
        reason = -1;
      }

      if (record_) {
        if (!is_recording || fp_ == nullptr) {
          throw std::runtime_error("Recording is not started");
        }
        fclose(fp_);
        is_recording = false;
      }
    }


    // update_time_stat(start, step_time_count_, step_time_);
    // step_time_count_++;

    // double step_time = 0;
    // if (done_) {
    //   step_time = step_time_;
    //   step_time_ = 0;
    //   step_time_count_ = 0;
    // }

    // if (done_) {
    //   update_time_stat(deck_name_[0], step_time_);
    //   update_time_stat(deck_name_[1], step_time_);
    //   step_time_ = 0;
    //   step_time_count_ = 0;
    // }
    // if (step_time_count_ % 3000 == 0) {
    //   fmt::println("Step time: {:.3f}", step_time_ * 1000);
    // }
    ret_reward_ = reward;
    ret_win_reason_ = reason;
  }

  using YGOProEnvSpec = EnvSpec<YGOProEnvFns>;
  using State =
      Dict<typename YGOProEnvSpec::StateKeys,
           typename SpecToTArray<typename YGOProEnvSpec::StateSpec::Values>::Type>;

  void WriteState(State &state) {
    float reward = ret_reward_;
    int win_reason = ret_win_reason_;
    int n_options = legal_actions_.size();
    state["reward"_] = reward;
    state["info:to_play"_] = int(to_play_);
    state["info:is_selfplay"_] = int(play_mode_ == kSelfPlay);
    state["info:win_reason"_] = win_reason;
    int anchor_seat = 0;
    if (!anchor_deck_.empty()) {
      if (deck_name_[0] == anchor_deck_) {
        anchor_seat = 1;
      } else if (deck_name_[1] == anchor_deck_) {
        anchor_seat = 2;
      }
    }
    state["info:anchor_seat"_] = anchor_seat;
    if (done_) {
      state["info:step_time"_][0] = 0;
      state["info:step_time"_][1] = 0;
      state["info:deck"_][0] = deck_names_ids_[deck_name_[0]];
      state["info:deck"_][1] = deck_names_ids_[deck_name_[1]];
    }

    if (n_options == 0) {
      state["info:num_options"_] = 1;
      state["obs:global_"_][22] = uint8_t(1);
      // if (step_count_ >= spec_.config["max_steps"_]) {
      //   fmt::println("Max steps reached return");
      // }
      return;
    }

    SpecInfos spec_infos;
    std::vector<int> loc_n_cards;

    if (spec_.config["oppo_info"_]) {
      _set_obs_g_cards(state["obs:cards_"_], to_play_);
      auto [spec_infos_, loc_n_cards_] = _set_obs_mask(state["obs:mask_"_], to_play_);
      spec_infos = spec_infos_;
      loc_n_cards = loc_n_cards_;
    } else {
      auto [spec_infos_, loc_n_cards_] = _set_obs_cards(state["obs:cards_"_], to_play_);
      spec_infos = spec_infos_;
      loc_n_cards = loc_n_cards_;
    }

    _set_obs_global(state["obs:global_"_], to_play_, loc_n_cards);

    if (n_options > max_options()) {
      throw std::runtime_error(fmt::format(
          "Legal action overflow: {} > max_options={} (msg={}, deck0={}, "
          "deck1={}, turn={}, phase={})",
          n_options, max_options(), msg_to_string(msg_), deck_name_[0],
          deck_name_[1], turn_count_, phase_to_string(current_phase_)));
    }

    n_options = legal_actions_.size();
    state["info:num_options"_] = n_options;

    for (int i = 0; i < n_options; ++i) {
      auto &action = legal_actions_[i];
      action.msg_ = msg_;
      const auto &spec = action.spec_;
      if (!spec.empty()) {
        const auto& spec_info = find_spec_info(spec_infos, spec);
        action.spec_index_ = spec_info.index;
        if (action.cid_ == 0) {
          action.cid_ = spec_info.cid;
        }
      }
    }

    _set_obs_actions(state["obs:actions_"_], legal_actions_);
    _set_obs_action_ir(
        state["obs:action_ir_"_], state["obs:action_single_refs_"_],
        state["obs:action_group_refs_"_],
        state["obs:action_group_mask_"_], spec_infos);
    _set_obs_selection(state["obs:selection_"_]);
    _set_obs_public_events(
        state["obs:public_events_"_],
        state["obs:public_event_refs_"_], spec_infos);

    // write history actions

    auto ha_p = to_play_ == 0 ? ha_p_1_ : ha_p_2_;
    auto &history_actions = to_play_ == 0 ? history_actions_1_ : history_actions_2_;

    int offset = n_history_actions_ - ha_p;
    int n_h_action_feats = history_actions.Shape()[1];

    state["obs:h_actions_"_].Assign(
      (uint8_t *)history_actions[ha_p].Data(), n_h_action_feats * offset);
    state["obs:h_actions_"_][offset].Assign(
      (uint8_t *)history_actions.Data(), n_h_action_feats * ha_p);
    
    for (int i = 0; i < n_history_actions_; ++i) {
      if (uint8_t(state["obs:h_actions_"_](i, 3)) == 0) {
        break;
      }
      // state["obs:h_actions_"_](i, 12) = static_cast<uint8_t>(uint8_t(state["obs:h_actions_"_](i, 12)) == to_play_);
      int turn_diff = std::min(16, turn_count_ - uint8_t(state["obs:h_actions_"_](i, 12)));
      state["obs:h_actions_"_](i, 12) = static_cast<uint8_t>(turn_diff);
    }
  }

private:
  using SpecInfos = ankerl::unordered_dense::map<std::string, SpecInfo>;

  void ensure_card_slot(int offset) const {
    const int capacity = max_cards() * 2;
    if (offset >= capacity) {
      throw std::runtime_error(fmt::format(
          "Card observation overflow: offset={} capacity={} "
          "(deck0={}, deck1={}, turn={}, phase={})",
          offset, capacity, deck_name_[0], deck_name_[1], turn_count_,
          phase_to_string(current_phase_)));
    }
  }

  std::tuple<SpecInfos, std::vector<int>> _set_obs_cards(TArray<uint8_t> &f_cards, PlayerId to_play) {
    SpecInfos spec_infos;
    std::vector<int> loc_n_cards;
    int offset = 0;
    for (auto pi = 0; pi < 2; pi++) {
      const PlayerId player = (to_play + pi) % 2;
      const bool opponent = pi == 1;
      std::vector<std::pair<uint8_t, bool>> configs = {
          {LOCATION_DECK, true},   {LOCATION_HAND, true},
          {LOCATION_MZONE, false}, {LOCATION_SZONE, false},
          {LOCATION_GRAVE, false}, {LOCATION_REMOVED, false},
          {LOCATION_EXTRA, true},
      };
      for (auto &[location, hidden_for_opponent] : configs) {
        if (opponent && hidden_for_opponent && revealed_.empty()) {
          auto n_cards = YGO_QueryFieldCount(pduel_, player, location);
          loc_n_cards.push_back(n_cards);
          for (auto i = 0; i < n_cards; i++) {
            ensure_card_slot(offset);
            f_cards(offset, 2) = location_to_id(location);
            f_cards(offset, 4) = 1;
            offset++;
          }
        } else {
          std::vector<Card> cards = get_cards_in_location(player, location);
          int n_cards = cards.size();
          loc_n_cards.push_back(n_cards);
          for (int i = 0; i < n_cards; ++i) {
            ensure_card_slot(offset);
            const auto &c = cards[i];
            auto spec = c.get_spec(opponent);
            bool hide = false;
            if (opponent) {
              hide = hidden_for_opponent || (c.position_ & POS_FACEDOWN);
              if (revealed_.find(spec) != revealed_.end()) {
                hide = false;
              }
            }
            CardId card_id = 0;
            if (!hide) {
              card_id = c_get_card_id(c.code_);
            }
            _set_obs_card_(f_cards, offset, c, hide, card_id);
            offset++;

            spec_infos[spec] = {static_cast<uint16_t>(offset), card_id};
          }
        }
      }
    }
    return {spec_infos, loc_n_cards};
  }

  void _set_obs_g_cards(TArray<uint8_t> &f_cards, PlayerId to_play) {
    int offset = 0;
    for (auto pi = 0; pi < 2; pi++) {
      const PlayerId player = (to_play + pi) % 2;
      std::vector<uint8_t> configs = {
          LOCATION_DECK, LOCATION_HAND, LOCATION_MZONE,
          LOCATION_SZONE, LOCATION_GRAVE, LOCATION_REMOVED,
          LOCATION_EXTRA,
      };
      for (auto location : configs) {
        std::vector<Card> cards = get_cards_in_location(player, location);
        int n_cards = cards.size();
        for (int i = 0; i < n_cards; ++i) {
          ensure_card_slot(offset);
          const auto &c = cards[i];
          CardId card_id = c_get_card_id(c.code_);
          _set_obs_card_(f_cards, offset, c, false, card_id, false);
          offset++;
        }
      }
    }
  }

  std::tuple<SpecInfos, std::vector<int>> _set_obs_mask(TArray<uint8_t> &mask, PlayerId to_play) {
    SpecInfos spec_infos;
    std::vector<int> loc_n_cards;
    int offset = 0;
    for (auto pi = 0; pi < 2; pi++) {
      const PlayerId player = (to_play + pi) % 2;
      const bool opponent = pi == 1;
      std::vector<std::pair<uint8_t, bool>> configs = {
          {LOCATION_DECK, true},   {LOCATION_HAND, true},
          {LOCATION_MZONE, false}, {LOCATION_SZONE, false},
          {LOCATION_GRAVE, false}, {LOCATION_REMOVED, false},
          {LOCATION_EXTRA, true},
      };
      for (auto &[location, hidden_for_opponent] : configs) {
        if (opponent && hidden_for_opponent && revealed_.empty()) {
          auto n_cards = YGO_QueryFieldCount(pduel_, player, location);
          loc_n_cards.push_back(n_cards);
          for (auto i = 0; i < n_cards; i++) {
            ensure_card_slot(offset);
            mask(offset, 1) = 1;
            mask(offset, 3) = 1;
            offset++;
          }
        } else {
          std::vector<Card> cards = get_cards_in_location(player, location);
          int n_cards = cards.size();
          loc_n_cards.push_back(n_cards);
          for (int i = 0; i < n_cards; ++i) {
            ensure_card_slot(offset);
            const auto &c = cards[i];
            auto spec = c.get_spec(opponent);
            bool hide = false;
            if (opponent) {
              hide = hidden_for_opponent || (c.position_ & POS_FACEDOWN);
              if (revealed_.find(spec) != revealed_.end()) {
                hide = false;
              }
            }
            CardId card_id = 0;
            if (!hide) {
              card_id = c_get_card_id(c.code_);
            }
            _set_obs_mask_(mask, offset, c, hide, card_id);
            offset++;

            spec_infos[spec] = {static_cast<uint16_t>(offset), card_id};
          }
        }
      }
    }
    return {spec_infos, loc_n_cards};
  }

  void _set_obs_card_(TArray<uint8_t> &f_cards, int offset, const Card &c,
                      bool hide, CardId card_id = 0, bool global = false) {
    // check offset exceeds max_cards
    uint8_t location = c.location_;
    bool overlay = location & LOCATION_OVERLAY;
    if (overlay) {
      location = location & 0x7f;
    }
    if (overlay) {
      hide = false;
    }

    if (!hide) {
      f_cards(offset, 0) = static_cast<uint8_t>(card_id >> 8);
      f_cards(offset, 1) = static_cast<uint8_t>(card_id & 0xff);
    }
    f_cards(offset, 2) = location_to_id(location);

    uint8_t seq = 0;
    if (location == LOCATION_MZONE || location == LOCATION_SZONE ||
        location == LOCATION_GRAVE) {
      seq = c.sequence_ + 1;
    }
    f_cards(offset, 3) = seq;
    f_cards(offset, 4) = global ? c.controler_ : ((c.controler_ != to_play_) ? 1 : 0);
    if (overlay) {
      f_cards(offset, 5) = position_to_id(POS_FACEUP);
      f_cards(offset, 6) = 1;
    } else {
      if (location == LOCATION_DECK || location == LOCATION_HAND || location == LOCATION_EXTRA) {
        if (hide || (c.position_ & POS_FACEDOWN)) {
          f_cards(offset, 5) = position_to_id(POS_FACEDOWN);
        }
        // else {
        //   fmt::println("location: {}, position: {}", location2str.at(location), position_to_string(c.position_));
        // }
      } else {
        f_cards(offset, 5) = position_to_id(c.position_);
      }
    }
    if (!hide) {
      f_cards(offset, 7) = attribute_to_id(c.attribute_);
      f_cards(offset, 8) = race_to_id(c.race_);
      f_cards(offset, 9) = c.level_;
      f_cards(offset, 10) = std::min(c.counter_, static_cast<uint32_t>(15));
      f_cards(offset, 11) = static_cast<uint8_t>((c.status_ & (STATUS_DISABLED | STATUS_FORBIDDEN)) != 0);
      auto [atk1, atk2] = float_transform(c.attack_);
      f_cards(offset, 12) = atk1;
      f_cards(offset, 13) = atk2;

      auto [def1, def2] = float_transform(c.defense_);
      f_cards(offset, 14) = def1;
      f_cards(offset, 15) = def2;

      auto type_ids = type_to_ids(c.type_);
      for (int j = 0; j < type_ids.size(); ++j) {
        f_cards(offset, 16 + j) = type_ids[j];
      }
    }
  }

  void _set_obs_mask_(TArray<uint8_t> &mask, int offset, const Card &c,
                      bool hide, CardId card_id = 0, bool global = false) {
    // check offset exceeds max_cards
    uint8_t location = c.location_;
    bool overlay = location & LOCATION_OVERLAY;
    if (overlay) {
      location = location & 0x7f;
    }
    if (overlay) {
      hide = false;
    }

    if (!hide) {
      if (card_id != 0) {
        mask(offset, 0) = 1;
      }
    }
    mask(offset, 1) = 1;

    if (location == LOCATION_MZONE || location == LOCATION_SZONE ||
        location == LOCATION_GRAVE) {
      mask(offset, 2) = 1;
    }
    mask(offset, 3) = 1;
    if (overlay) {
      mask(offset, 4) = 1;
      mask(offset, 5) = 1;
    } else {
      if (location == LOCATION_DECK || location == LOCATION_HAND || location == LOCATION_EXTRA) {
        if (hide || (c.position_ & POS_FACEDOWN)) {
          mask(offset, 4) = 1;
        }
      } else {
        mask(offset, 4) = 1;
      }
    }
    if (!hide) {
      mask(offset, 6) = 1;
      mask(offset, 7) = 1;
      mask(offset, 8) = 1;
      mask(offset, 9) = 1;
      mask(offset, 10) = 1;
      mask(offset, 11) = 1;
      mask(offset, 12) = 1;
      mask(offset, 13) = 1;
    }
  }

  void _set_obs_global(TArray<uint8_t> &feat, PlayerId player, const std::vector<int> &loc_n_cards) {
    uint8_t me = player;
    uint8_t op = 1 - player;

    auto [me_lp_1, me_lp_2] = float_transform(lp_[me]);
    feat(0) = me_lp_1;
    feat(1) = me_lp_2;

    auto [op_lp_1, op_lp_2] = float_transform(lp_[op]);
    feat(2) = op_lp_1;
    feat(3) = op_lp_2;

    feat(4) = std::min(turn_count_, 16);
    feat(5) = phase_to_id(current_phase_);
    feat(6) = (me == 0) ? 1 : 0;
    feat(7) = (me == tp_) ? 1 : 0;

    for (int i = 0; i < loc_n_cards.size(); i++) {
      feat(8 + i) = static_cast<uint8_t>(loc_n_cards[i]);
    }
  }

  const SpecInfo& find_spec_info(SpecInfos &spec_infos, const std::string &spec) {
    auto it = spec_infos.find(spec);
    if (it == spec_infos.end()) {
      // TODO(2): find the root cause
      // print spec2index
      show_deck(0);
      show_deck(1);
      show_buffer();
      show_turn();
      fmt::println("MS: idx: {}, mode: {}, min: {}, max: {}, must: {}, specs: {}, combs: {}", ms_idx_, ms_mode_, ms_min_, ms_max_, ms_must_, ms_specs_, ms_combs_);
      fmt::println("Spec: {}, Spec2index:", spec);
      for (auto &[k, v] : spec_infos) {
        fmt::print("{}: {} {}, ", k, v.index, v.cid);
      }
      fmt::print("\n");
      // throw std::runtime_error("Spec not found: " + spec);
      spec_infos[spec] = {0, 0};
      return spec_infos[spec];
    }
    return it->second;
  }

  void _set_obs_action_spec(
    TArray<uint8_t> &feat, int i, int idx) {
    feat(i, 0) = static_cast<uint8_t>(idx);
  }

  void _set_obs_action_card_id(
    TArray<uint8_t> &feat, int i, CardId cid) {
    feat(i, 1) = static_cast<uint8_t>(cid >> 8);
    feat(i, 2) = static_cast<uint8_t>(cid & 0xff);
  }

  void _set_obs_action_msg(TArray<uint8_t> &feat, int i, int msg) {
    feat(i, 3) = msg_to_id(msg);
  }

  void _set_obs_action_act(TArray<uint8_t> &feat, int i, ActionAct act) {
    feat(i, 4) = static_cast<uint8_t>(act);
  }

  void _set_obs_action_finish(TArray<uint8_t> &feat, int i) {
    feat(i, 5) = 1;
  }

  uint8_t action_effect_id(int effect) const {
    if (effect == -1) {
      return 0;
    }
    if (effect == 0) {
      return 1;
    }
    if (effect >= CARD_EFFECT_OFFSET) {
      return static_cast<uint8_t>(
          std::min(255, effect - CARD_EFFECT_OFFSET + 2));
    }
    return system_string_to_id(effect);
  }

  void _set_obs_action_effect(TArray<uint8_t> &feat, int i, int effect) {
    // 0: None
    // 1: default
    // 2-15: card effect
    // 16+: system
    feat(i, 6) = action_effect_id(effect);
  }

  void _set_obs_action_phase(TArray<uint8_t> &feat, int i, ActionPhase phase){
    feat(i, 7) = static_cast<uint8_t>(phase);
  }

  void _set_obs_action_position(TArray<uint8_t> &feat, int i, uint8_t position) {
    feat(i, 8) = position_to_id(position);
  }

  void _set_obs_action_number(TArray<uint8_t> &feat, int i, uint8_t number) {
    feat(i, 9) = number;
  }

  void _set_obs_action_place(TArray<uint8_t> &feat, int i, ActionPlace place) {
    feat(i, 10) = static_cast<uint8_t>(place);
  }

  void _set_obs_action_attrib(TArray<uint8_t> &feat, int i, uint8_t attrib) {
    feat(i, 11) = attribute_to_id(attrib);
  }

  void _set_obs_action(TArray<uint8_t> &feat, int i, const LegalAction &action) {
    auto msg = action.msg_;
    _set_obs_action_msg(feat, i, msg);
    _set_obs_action_card_id(feat, i, action.cid_);
    if (msg == MSG_SELECT_CARD || msg == MSG_SELECT_TRIBUTE ||
        msg == MSG_SELECT_SUM || msg == MSG_SELECT_UNSELECT_CARD) {
      if (action.finish_) {
        _set_obs_action_finish(feat, i);
      } else {
        _set_obs_action_spec(feat, i, action.spec_index_);
      }
    } else if (msg == MSG_SELECT_POSITION) {
      _set_obs_action_position(feat, i, action.position_);
    } else if (msg == MSG_SELECT_EFFECTYN) {
      _set_obs_action_spec(feat, i, action.spec_index_);
      _set_obs_action_act(feat, i, action.act_);
      _set_obs_action_effect(feat, i, action.effect_);
    } else if (msg == MSG_SELECT_YESNO || msg == MSG_SELECT_OPTION) {
      _set_obs_action_act(feat, i, action.act_);
      _set_obs_action_effect(feat, i, action.effect_);
    } else if (
      msg == MSG_SELECT_BATTLECMD ||
      msg == MSG_SELECT_IDLECMD ||
      msg == MSG_SELECT_CHAIN) {
      _set_obs_action_phase(feat, i, action.phase_);
      _set_obs_action_spec(feat, i, action.spec_index_);
      _set_obs_action_act(feat, i, action.act_);
      _set_obs_action_effect(feat, i, action.effect_);
    } else if (msg == MSG_SELECT_PLACE || msg_ == MSG_SELECT_DISFIELD) {
      _set_obs_action_place(feat, i, action.place_);
    } else if (msg == MSG_ANNOUNCE_CARD) {
      // card id, already set
    } else if (msg == MSG_ANNOUNCE_ATTRIB) {
      _set_obs_action_attrib(feat, i, action.attribute_);
    } else if (msg == MSG_ANNOUNCE_NUMBER) {
      _set_obs_action_number(feat, i, action.number_);
    } else if (msg == MSG_ANNOUNCE_RACE) {
      feat(i, 6) = static_cast<uint8_t>(
          ACTION_RACE_EFFECT_OFFSET + action.number_);
    } else {
      throw std::runtime_error("Unsupported message " + msg_to_string(msg));
    }
  }

  CardId spec_to_card_id(const std::string &spec, PlayerId player) {
    int offset = 0;
    bool opponent = false;
    if (spec[0] == 'o') {
      player = 1 - player;
      opponent = true;
      offset++;
    }
    auto [loc, seq, pos] = spec_to_ls(spec.substr(offset));
    if (opponent) {
      const bool hidden_for_opponent =
          loc != LOCATION_MZONE && loc != LOCATION_SZONE &&
          loc != LOCATION_GRAVE && loc != LOCATION_REMOVED;
      const bool revealed = revealed_.find(spec) != revealed_.end();
      if (hidden_for_opponent && !revealed) {
        return 0;
      }
      Card c = get_card(player, loc, seq);
      const bool hide =
          !revealed &&
          (hidden_for_opponent || (c.position_ & POS_FACEDOWN));
      return hide ? 0 : c_get_card_id(c.code_);
    }
    return c_get_card_id(get_card_code(player, loc, seq));
  }

  void _set_obs_actions(TArray<uint8_t> &feat, const std::vector<LegalAction> &actions) {
    for (int i = 0; i < actions.size(); ++i) {
      _set_obs_action(feat, i, actions[i]);
    }
  }

  uint8_t lookup_spec_ref(
      const SpecInfos &spec_infos, const std::string &spec) const {
    if (spec.empty()) {
      return 0;
    }
    const auto it = spec_infos.find(spec);
    if (it == spec_infos.end()) {
      return 0;
    }
    if (it->second.index > 255) {
      throw std::runtime_error("Card reference exceeds uint8 range");
    }
    return static_cast<uint8_t>(it->second.index);
  }

  CardId lookup_spec_card_id(
      const SpecInfos &spec_infos, const std::string &spec) const {
    if (spec.empty()) {
      return 0;
    }
    const auto it = spec_infos.find(spec);
    return it == spec_infos.end() ? 0 : it->second.cid;
  }

  void _set_obs_action_ir(
      TArray<uint8_t> &ir, TArray<uint8_t> &single_refs,
      TArray<uint8_t> &group_refs, TArray<uint8_t> &group_mask,
      const SpecInfos &spec_infos) {
    const bool finishable = std::any_of(
        legal_actions_.begin(), legal_actions_.end(),
        [](const LegalAction &action) { return action.finish_; });
    const bool cancelable = std::any_of(
        legal_actions_.begin(), legal_actions_.end(),
        [](const LegalAction &action) {
          return action.act_ == ActionAct::Cancel;
        });
    const auto role = selection_role_for_msg(msg_);
    const bool active_selection = ms_idx_ >= 0;
    const int stage = active_selection ? ms_idx_ : 0;
    const int selected_count = active_selection
        ? static_cast<int>(ms_r_idxs_.size())
        : 0;
    const int minimum = active_selection ? ms_min_ : 0;
    const int maximum = active_selection ? ms_max_ : 0;
    const int must = active_selection ? ms_must_ : 0;
    const bool mandatory =
        active_selection && selected_count < std::max(minimum, must);

    std::vector<uint8_t> selected_refs;
    selected_refs.reserve(kMaxActionRoleMembers);
    if (active_selection) {
      for (const auto selected_index : ms_r_idxs_) {
        if (selected_index < 0 ||
            selected_index >= static_cast<int>(ms_specs_.size())) {
          continue;
        }
        const auto ref = lookup_spec_ref(
            spec_infos, ms_specs_[selected_index]);
        if (ref != 0) {
          selected_refs.push_back(ref);
        }
        if (selected_refs.size() == kMaxActionRoleMembers) {
          break;
        }
      }
    }

    for (int i = 0; i < static_cast<int>(legal_actions_.size()); ++i) {
      const auto &action = legal_actions_[i];
      const bool candidate_action =
          is_candidate_selection_msg(action.msg_) &&
          !action.finish_ && action.act_ != ActionAct::Cancel;

      uint8_t source_ref = 0;
      uint8_t candidate_ref = 0;
      CardId source_cid = 0;
      int source_effect = action.effect_;
      uint8_t binding_confidence = 0;

      if (candidate_action) {
        candidate_ref = static_cast<uint8_t>(
            std::min(action.spec_index_, 255));
      } else if (!action.spec_.empty()) {
        source_ref = static_cast<uint8_t>(
            std::min(action.spec_index_, 255));
        source_cid = action.cid_;
        binding_confidence = source_ref == 0 ? 0 : 2;
      }

      if (source_ref == 0 && !pending_source_spec_.empty() &&
          !is_source_action_msg(action.msg_)) {
        const auto observer_spec = spec_for_observer(
            pending_source_spec_, pending_source_spec_player_, to_play_);
        source_ref = lookup_spec_ref(spec_infos, observer_spec);
        source_cid = lookup_spec_card_id(spec_infos, observer_spec);
        if (source_cid == 0 &&
            pending_source_spec_player_ == to_play_) {
          source_cid = pending_source_cid_;
        }
        source_effect = pending_source_effect_;
        binding_confidence = source_ref == 0 ? 0 : 1;
      }

      single_refs(i, 0) = source_ref;
      single_refs(i, 1) = candidate_ref;
      if (action.act_ == ActionAct::Attack ||
          action.act_ == ActionAct::DirectAttack) {
        single_refs(i, 2) = source_ref;
      }

      for (int member = 0;
           member < static_cast<int>(selected_refs.size()); ++member) {
        group_refs(i, 4, member) = selected_refs[member];
        group_mask(i, 4, member) = 1;
        if (role == SelectionRole::Material) {
          group_refs(i, 2, member) = selected_refs[member];
          group_mask(i, 2, member) = 1;
        } else if (role == SelectionRole::Tribute) {
          group_refs(i, 3, member) = selected_refs[member];
          group_mask(i, 3, member) = 1;
        } else if (role == SelectionRole::Target) {
          group_refs(i, 0, member) = selected_refs[member];
          group_mask(i, 0, member) = 1;
        } else if (role == SelectionRole::Cost) {
          group_refs(i, 1, member) = selected_refs[member];
          group_mask(i, 1, member) = 1;
        }
      }

      ir(i, 0) = msg_to_id(action.msg_);
      ir(i, 1) = static_cast<uint8_t>(action.act_);
      ir(i, 2) = static_cast<uint8_t>(action.phase_);
      ir(i, 3) = static_cast<uint8_t>(role);
      ir(i, 4) = static_cast<uint8_t>(std::min(stage, 15));
      ir(i, 5) = static_cast<uint8_t>(action.finish_);
      ir(i, 6) = static_cast<uint8_t>(
          action.act_ == ActionAct::Cancel);
      ir(i, 7) = static_cast<uint8_t>(mandatory);
      ir(i, 8) = static_cast<uint8_t>(finishable);
      ir(i, 9) = static_cast<uint8_t>(cancelable);
      ir(i, 10) = static_cast<uint8_t>(std::min(minimum, 255));
      ir(i, 11) = static_cast<uint8_t>(std::min(maximum, 255));
      ir(i, 12) = static_cast<uint8_t>(
          std::min(selected_count, 255));
      ir(i, 13) = static_cast<uint8_t>(std::min(must, 255));
      ir(i, 14) = action_effect_id(source_effect);
      ir(i, 15) = static_cast<uint8_t>(source_cid >> 8);
      ir(i, 16) = static_cast<uint8_t>(source_cid & 0xff);
      ir(i, 17) = binding_confidence;
      ir(i, 18) = position_to_id(action.position_);
      ir(i, 19) = static_cast<uint8_t>(action.place_);
      ir(i, 20) = action.number_;
      ir(i, 21) = attribute_to_id(action.attribute_);
      ir(i, 22) = static_cast<uint8_t>(std::min(i + 1, 255));
      ir(i, 23) = static_cast<uint8_t>(
          active_selection ? ms_mode_ : 0);
    }
  }

  void _set_obs_selection(TArray<uint8_t> &selection) {
    const bool finishable = std::any_of(
        legal_actions_.begin(), legal_actions_.end(),
        [](const LegalAction &action) { return action.finish_; });
    const bool cancelable = std::any_of(
        legal_actions_.begin(), legal_actions_.end(),
        [](const LegalAction &action) {
          return action.act_ == ActionAct::Cancel;
        });
    const bool active_selection = ms_idx_ >= 0;
    const int selected_count = active_selection
        ? static_cast<int>(ms_r_idxs_.size())
        : 0;
    const int minimum = active_selection ? ms_min_ : 0;
    const int maximum = active_selection ? ms_max_ : 0;
    const int must = active_selection ? ms_must_ : 0;
    selection(0) = msg_to_id(msg_);
    selection(1) = static_cast<uint8_t>(selection_role_for_msg(msg_));
    selection(2) = static_cast<uint8_t>(
        active_selection ? ms_idx_ : 0);
    selection(3) = static_cast<uint8_t>(std::min(minimum, 255));
    selection(4) = static_cast<uint8_t>(std::min(maximum, 255));
    selection(5) = static_cast<uint8_t>(std::min(must, 255));
    selection(6) = static_cast<uint8_t>(
        std::min(selected_count, 255));
    selection(7) = static_cast<uint8_t>(finishable);
    selection(8) = static_cast<uint8_t>(cancelable);
    selection(9) = static_cast<uint8_t>(
        active_selection && selected_count < std::max(minimum, must));
    selection(10) = static_cast<uint8_t>(
        active_selection ? ms_mode_ : 0);
    selection(11) = static_cast<uint8_t>(
        std::min(static_cast<int>(legal_actions_.size()), 255));
  }

  void _set_obs_public_events(
      TArray<uint8_t> &events, TArray<uint8_t> &event_refs,
      const SpecInfos &spec_infos) {
    const int count = std::min(
        static_cast<int>(public_events_.size()), n_history_actions_);
    for (int i = 0; i < count; ++i) {
      const auto &event = public_events_[i];
      const auto source_spec = spec_for_observer(
          event.source_spec, event.source_spec_player, to_play_);
      const auto candidate_spec = spec_for_observer(
          event.candidate_spec, event.candidate_spec_player, to_play_);
      const auto source_ref = lookup_spec_ref(spec_infos, source_spec);
      const auto candidate_ref = lookup_spec_ref(
          spec_infos, candidate_spec);

      events(i, 0) = 1;
      events(i, 1) = event.actor == to_play_ ? 1 : 2;
      events(i, 2) = msg_to_id(event.msg);
      events(i, 3) = static_cast<uint8_t>(event.act);
      events(i, 4) = static_cast<uint8_t>(event.action_phase);
      events(i, 5) = static_cast<uint8_t>(event.selection_role);
      events(i, 6) = static_cast<uint8_t>(
          std::min(16, std::max(0, turn_count_ - event.turn)));
      events(i, 7) = phase_to_id(event.phase);
      events(i, 8) = static_cast<uint8_t>(event.finish);
      events(i, 9) = static_cast<uint8_t>(event.cancel);
      if (event.payload_size != 0) {
        events(i, 10) = event.payload_size;
        for (int payload_index = 0;
             payload_index < event.payload_size; ++payload_index) {
          events(i, 11 + payload_index) =
              event.payload[payload_index];
        }
      } else {
        events(i, 10) = action_effect_id(event.effect);
        events(i, 11) = static_cast<uint8_t>(
            std::min(event.selection_stage, 15));
        events(i, 12) = static_cast<uint8_t>(
            std::min(event.selected_count, 255));
        events(i, 13) = static_cast<uint8_t>(source_ref != 0);
        events(i, 14) = static_cast<uint8_t>(candidate_ref != 0);
        events(i, 15) = event.choice;
      }
      event_refs(i, 0) = source_ref;
      event_refs(i, 1) = candidate_ref;
      if (event.act == ActionAct::Attack ||
          event.act == ActionAct::DirectAttack) {
        event_refs(i, 2) = source_ref;
      }
    }
  }


  void str_to_uint16(const char* src, uint16_t* dest) {
      for (int i = 0; i < strlen(src); i += 1) {
        dest[i] = src[i];
      }

      // Add null terminator
      dest[strlen(src) + 1] = '\0';
  }

  void ReplayWriteInt8(int8_t value) {
    fwrite(&value, sizeof(value), 1, fp_);
  }

  void ReplayWriteInt32(int32_t value) {
    fwrite(&value, sizeof(value), 1, fp_);
  }

  // ygopro-core API
  intptr_t YGO_CreateDuel(uint32_t seed) {
    // Replay headers store the outer seed; standard YGOPro derives the core
    // seed with one MT draw before create_duel.
    std::mt19937 rnd(seed);
    return create_duel(rnd());
  }

  void YGO_SetPlayerInfo(intptr_t pduel, int32 playerid, int32 lp, int32 startcount, int32 drawcount) const {
    set_player_info(pduel, playerid, lp, startcount, drawcount);
  }

  void YGO_NewCard(intptr_t pduel, uint32 code, uint8 owner, uint8 playerid, uint8 location, uint8 sequence, uint8 position) const {
    new_card(pduel, code, owner, playerid, location, sequence, position);
  }

  void YGO_StartDuel(intptr_t pduel, int32 options) const {
    start_duel(pduel, options);
  }

  void YGO_EndDuel(intptr_t pduel) const {
    // end_duel(pduel);
    duel* pd = (duel*)pduel;
    delete pd;
  }

  int32 YGO_GetMessage(intptr_t pduel, byte* buf) {
    return get_message(pduel, buf);
  }

  uint32 YGO_Process(intptr_t pduel) {
    return process(pduel);
  }

  int32 YGO_QueryCard(intptr_t pduel, uint8 playerid, uint8 location, uint8 sequence, int32 query_flag, byte* buf) {
    return query_card(pduel, playerid, location, sequence, query_flag, buf, 0);
  }

  int32 YGO_QueryFieldCount(intptr_t pduel, uint8 playerid, uint8 location) {
    return query_field_count(pduel, playerid, location);
  }

  int32 YGO_QueryFieldCard(intptr_t pduel, uint8 playerid, uint8 location, uint32 query_flag, byte* buf) {
    return query_field_card(pduel, playerid, location, query_flag, buf, 0);
  }

  void YGO_SetResponsei(intptr_t pduel, int32 value) {
    if (record_) {
      ReplayWriteInt8(4);
      ReplayWriteInt32(value);
    }
    set_responsei(pduel, value);
  }

  void YGO_SetResponseb(intptr_t pduel, byte* buf) {
    if (record_) {
      switch (msg_) {
        case MSG_SORT_CARD:
          ReplayWriteInt8(1);
          fwrite(buf, 1, 1, fp_);
          break;
        case MSG_SELECT_COUNTER:
          ReplayWriteInt8(2 * n_counters_);
          fwrite(buf, 2 * n_counters_, 1, fp_);
          break;
        case MSG_SELECT_PLACE:
        case MSG_SELECT_DISFIELD:
          ReplayWriteInt8(3);
          fwrite(buf, 3, 1, fp_);
          break;
        default:
          ReplayWriteInt8(buf[0] + 1);
          fwrite(buf, buf[0] + 1, 1, fp_);
          break;
      }
    }
    set_responseb(pduel, buf);
  }

  // ygopro-core API

  void show_decision(int idx) {
    std::string s;
    const auto& a = legal_actions_[idx];
    if (!a.spec_.empty()) {
      s = a.spec_;
    } else if (a.place_ != ActionPlace::None) {
      s = action_place_to_string(a.place_);
    } else if (a.position_ != 0) {
      s = position_to_string(a.position_);
    } else {
      s = fmt::format("{}", a);
    }
    fmt::print("Player {} chose \"{}\" in {}\n", to_play_, s, legal_actions_);
  }

  std::tuple<std::vector<CardCode>, std::vector<CardCode>, std::string>
  load_deck(
    intptr_t pduel, PlayerId player, const std::string &deck_name,
    std::mt19937& gen, bool shuffle = true) const {
    std::vector<CardCode> main_deck = main_decks_.at(deck_name);
    std::vector<CardCode> extra_deck = extra_decks_.at(deck_name);

    if (verbose_) {
      fmt::println("{} {}: {}, main({}), extra({})", player, nickname_[player],
        deck_name, main_deck.size(), extra_deck.size());
    }

    if (shuffle) {
      std::shuffle(main_deck.begin(), main_deck.end(), gen);
    }

    // Keep the insertion order aligned with this environment's replay writer.
    for (int i = 0; i < main_deck.size(); i++) {
      YGO_NewCard(pduel, main_deck[i], player, player, LOCATION_DECK, 0,
               POS_FACEDOWN_DEFENSE);
    }

    // add extra deck in reverse order following ygopro
    for (int i = int(extra_deck.size()) - 1; i >= 0; --i) {
      YGO_NewCard(pduel, extra_deck[i], player, player, LOCATION_EXTRA, 0,
               POS_FACEDOWN_DEFENSE);
    }

    return {main_deck, extra_deck, deck_name};
  }

  void next() {
    while (duel_started_) {
      if (eng_flag_ == PROCESSOR_END) {
        break;
      }
      uint32_t res = YGO_Process(pduel_);
      dl_ = res & PROCESSOR_BUFFER_LEN;
      eng_flag_ = res & PROCESSOR_FLAG;

      if (dl_ == 0) {
        continue;
      }
      YGO_GetMessage(pduel_, data_);
      dp_ = 0;
      while ((dp_ != dl_) || (ms_idx_ != -1)) {
        if (ms_idx_ != -1) {
          handle_multi_select();
        } else {
          handle_message();
          if (legal_actions_.empty()) {
            continue;
          }
        }
        if ((play_mode_ == kSelfPlay) || (to_play_ == ai_player_)) {
          if (legal_actions_.size() == 1) {
            callback_(0);
            auto la = legal_actions_[0];
            la.msg_ = msg_;
            if (la.cid_ == 0 && !la.spec_.empty()) {
              la.cid_ = spec_to_card_id(la.spec_, to_play_);
            }
            update_history_actions(to_play_, la);
            record_public_event(to_play_, la);
            if (verbose_) {
              show_decision(0);
            }
          } else {
            return;
          }
        } else {
          auto idx = players_[to_play_]->think(legal_actions_);
          callback_(idx);
          if (verbose_) {
            show_decision(idx);
          }
        }
      }
    }
    done_ = true;
    legal_actions_.clear();
  }

  uint8_t read_u8() { return data_[dp_++]; }

  uint16_t read_u16() {
    uint16_t v = *reinterpret_cast<uint16_t *>(data_ + dp_);
    dp_ += 2;
    return v;
  }

  uint32 read_u32() {
    uint32 v = *reinterpret_cast<uint32_t *>(data_ + dp_);
    dp_ += 4;
    return v;
  }

  uint32 q_read_u8() {
    uint8_t v = *reinterpret_cast<uint8_t *>(query_buf_ + qdp_);
    qdp_ += 1;
    return v;
  }

  uint32 q_read_u32() {
    uint32_t v = *reinterpret_cast<uint32_t *>(query_buf_ + qdp_);
    qdp_ += 4;
    return v;
  }

  CardCode get_card_code(PlayerId player, uint8_t loc, uint8_t seq) {
    int32_t flags = QUERY_CODE;
    int32_t bl = YGO_QueryCard(pduel_, player, loc, seq, flags, query_buf_);
    qdp_ = 0;
    if (bl <= 0) {
      throw std::runtime_error("[get_card_code] Invalid card");
    }
    qdp_ += 8;
    return q_read_u32();
  }

  Card get_card(PlayerId player, uint8_t loc, uint8_t seq) {
    int32_t flags = QUERY_CODE | QUERY_ATTACK | QUERY_DEFENSE | QUERY_POSITION |
                    QUERY_LEVEL | QUERY_RANK | QUERY_LSCALE | QUERY_RSCALE |
                    QUERY_LINK;
    int32_t bl = YGO_QueryCard(pduel_, player, loc, seq, flags, query_buf_);
    qdp_ = 0;
    if (bl <= 0) {
      show_deck(0);
      show_deck(1);
      show_turn();
      show_buffer();
      auto s = fmt::format("[get_card] Invalid card (bl <= 0), player: {}, loc: {}, seq: {}", player, loc, seq);
      throw std::runtime_error(s);
    }
    uint32_t f = q_read_u32();
    if (f == LEN_EMPTY) {
      return Card();
    }
    f = q_read_u32();
    CardCode code = q_read_u32();
    Card c = c_get_card(code);
    uint32_t position = q_read_u32();
    c.set_location(position);
    uint32_t level = q_read_u32();
    if ((level & 0xff) > 0) {
      c.level_ = level & 0xff;
    }
    uint32_t rank = q_read_u32();
    if ((rank & 0xff) > 0) {
      c.level_ = rank & 0xff;
    }
    c.attack_ = q_read_u32();
    c.defense_ = q_read_u32();
    c.lscale_ = q_read_u32();
    c.rscale_ = q_read_u32();
    uint32_t link = q_read_u32();
    uint32_t link_marker = q_read_u32();
    if ((link & 0xff) > 0) {
      c.level_ = link & 0xff;
    }
    if (link_marker > 0) {
      c.defense_ = link_marker;
    }
    return c;
  }

  std::vector<Card> get_cards_in_location(PlayerId player, uint8_t loc) {
    int32_t flags = QUERY_CODE | QUERY_POSITION | QUERY_LEVEL | QUERY_RANK |
                    QUERY_ATTACK | QUERY_DEFENSE | QUERY_EQUIP_CARD |
                    QUERY_OVERLAY_CARD | QUERY_COUNTERS | QUERY_STATUS |
                    QUERY_LSCALE | QUERY_RSCALE | QUERY_LINK;
    int32_t bl = YGO_QueryFieldCard(pduel_, player, loc, flags, query_buf_);
    qdp_ = 0;
    std::vector<Card> cards;
    while (true) {
      if (qdp_ >= bl) {
        break;
      }
      uint32_t f = q_read_u32();
      if (f == LEN_EMPTY) {
        continue;
        ;
      }
      f = q_read_u32();
      CardCode code = q_read_u32();
      Card c = c_get_card(code);

      uint8_t controller = q_read_u8();
      uint8_t location = q_read_u8();
      uint8_t sequence = q_read_u8();
      uint8_t position = q_read_u8();
      c.controler_ = controller;
      c.location_ = location;
      c.sequence_ = sequence;
      c.position_ = position;

      uint32_t level = q_read_u32();
      if ((level & 0xff) > 0) {
        c.level_ = level & 0xff;
      }
      uint32_t rank = q_read_u32();
      if ((rank & 0xff) > 0) {
        c.level_ = rank & 0xff;
      }
      c.attack_ = q_read_u32();
      c.defense_ = q_read_u32();

      // TODO(2): equip_target
      if (f & QUERY_EQUIP_CARD) {
        q_read_u32();
      }

      uint32_t n_xyz = q_read_u32();
      for (int i = 0; i < n_xyz; ++i) {
        auto code = q_read_u32();
        Card c_ = c_get_card(code);
        c_.controler_ = controller;
        c_.location_ = location | LOCATION_OVERLAY;
        c_.sequence_ = sequence;
        c_.position_ = i;
        cards.push_back(c_);
      }

      // TODO(2): counters
      uint32_t n_counters = q_read_u32();
      for (int i = 0; i < n_counters; ++i) {
        if (i == 0) {
          c.counter_ = q_read_u32();
        }
        else {
          q_read_u32();
        }
      }

      c.status_ = q_read_u32();
      c.lscale_ = q_read_u32();
      c.rscale_ = q_read_u32();

      uint32_t link = q_read_u32();
      uint32_t link_marker = q_read_u32();
      if ((link & 0xff) > 0) {
        c.level_ = link & 0xff;
      }
      if (link_marker > 0) {
        c.defense_ = link_marker;
      }
      cards.push_back(c);
    }
    return cards;
  }

  std::vector<Card> read_cardlist(bool extra = false, bool extra8 = false) {
    std::vector<Card> cards;
    auto count = read_u8();
    cards.reserve(count);
    for (int i = 0; i < count; ++i) {
      auto code = read_u32();
      auto controller = read_u8();
      auto loc = read_u8();
      auto seq = read_u8();
      auto card = get_card(controller, loc, seq);
      if (extra) {
        if (extra8) {
          card.data_ = read_u8();
        } else {
          card.data_ = read_u32();
        }
      }
      cards.push_back(card);
    }
    return cards;
  }

  std::vector<IdleCardSpec> read_cardlist_spec(PlayerId player, bool extra = false, bool extra8 = false) {
    std::vector<IdleCardSpec> card_specs;
    auto count = read_u8();
    card_specs.reserve(count);
    for (int i = 0; i < count; ++i) {
      CardCode code = read_u32();
      auto controller = read_u8();
      auto loc = read_u8();
      auto seq = read_u8();
      uint32_t data = 0;
      if (extra) {
        if (extra8) {
          data = read_u8();
        } else {
          data = read_u32();
        }
      }
      card_specs.push_back({code, ls_to_spec(loc, seq, 0, player != controller), data});
    }
    return card_specs;
  }

  std::tuple<CardCode, int> unpack_desc(CardCode code, uint32_t desc) {
    if (desc < DESCRIPTION_LIMIT) {
      return {0, desc};
    }
    CardCode code_ = desc >> 4;
    int idx = desc & 0xf;
    if (idx < 0 || idx >= 14) {
      fmt::print("Code: {}, Code_: {}, Desc: {}\n", code, code_, desc);
      show_deck(0);
      show_deck(1);
      show_buffer();
      show_turn();
      throw std::runtime_error("Invalid effect index: " + std::to_string(idx));
    }
    return {code_, idx + CARD_EFFECT_OFFSET};
  }

  std::string cardlist_info_for_player(const Card &card, PlayerId pl) {
    std::string spec = card.get_spec(pl);
    if (card.location_ == LOCATION_DECK) {
      spec = "deck";
    }
    if ((card.controler_ != pl) && (card.position_ & POS_FACEDOWN)) {
      return position_to_string(card.position_) + "card (" + spec + ")";
    }
    return card.name_ + " (" + spec + ")";
  }

  // This function does the following:
  // 1. read msg_ from data_ and update dp_
  // 2. (optional) print information if verbose_ is true
  // 3. update to_play_ and options_ if need action
  void handle_message() {
    msg_ = int(data_[dp_++]);
    legal_actions_ = {};

    if (verbose_) {
      fmt::println("Message {}, length {}, dp {}", msg_to_string(msg_), dl_, dp_);
    }

    if (msg_ == MSG_DRAW) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto player = read_u8();
      auto drawed = read_u8();
      std::vector<uint32> codes;
      for (int i = 0; i < drawed; ++i) {
        uint32 code = read_u32();
        codes.push_back(code & 0x7fffffff);
      }
      const auto &pl = players_[player];
      pl->notify(fmt::format("Drew {} cards:", drawed));
      for (int i = 0; i < drawed; ++i) {
        const auto &c = c_get_card(codes[i]);
        pl->notify(fmt::format("{}: {}", i + 1, c.name_));
      }
      const auto &op = players_[1 - player];
      op->notify(fmt::format("Opponent drew {} cards.", drawed));
    } else if (msg_ == MSG_TOSS_COIN || msg_ == MSG_TOSS_DICE) {
      const auto player = read_u8();
      const int count = read_u8();
      if (count < 1 || count > 5 || dp_ + count > dl_) {
        throw std::runtime_error(fmt::format(
            "Invalid {} payload: count={}, remaining={}",
            msg_to_string(msg_), count, dl_ - dp_));
      }
      std::vector<uint8_t> results;
      results.reserve(count);
      for (int i = 0; i < count; ++i) {
        const auto result = read_u8();
        const bool valid = msg_ == MSG_TOSS_COIN
            ? result <= 1
            : result >= 1 && result <= 6;
        if (!valid) {
          throw std::runtime_error(fmt::format(
              "Invalid {} result: {}", msg_to_string(msg_), result));
        }
        results.push_back(result);
      }
      record_protocol_event(player, msg_, results);
      if (verbose_) {
        fmt::println(
            "{} player={} count={} results={}",
            msg_to_string(msg_), player, count, results);
      }
      return;
    } else if (msg_ == MSG_NEW_TURN) {
      tp_ = int(read_u8());
      turn_count_++;
      if (!verbose_) {
        return;
      }
      auto& player = players_[tp_];
      player->notify("Your turn.");
      players_[1 - tp_]->notify(fmt::format("{}'s turn.", player->nickname_));
    } else if (msg_ == MSG_NEW_PHASE) {
      current_phase_ = int(read_u16());
      if (!verbose_) {
        return;
      }
      auto phase_str = phase_to_string(current_phase_);
      for (int i = 0; i < 2; ++i) {
        players_[i]->notify(fmt::format("Entering {} phase.", phase_str));
      }
    } else if (msg_ == MSG_FIELD_DISABLED) {
      const int payload_size = dl_ - dp_;
      if (payload_size < static_cast<int>(sizeof(uint32_t))) {
        throw std::runtime_error(fmt::format(
            "Truncated field_disabled payload: {}", payload_size));
      }
      disabled_field_ = read_u32();
      if (verbose_) {
        fmt::println(
            "Disabled field mask updated: player0=0x{:04x}, "
            "player1=0x{:04x}",
            disabled_field_ & 0xffff, disabled_field_ >> 16);
      }
      return;
    } else if (msg_ == MSG_MOVE) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      CardCode code = read_u32();
      uint32_t location = read_u32();
      uint32_t newloc = read_u32();
      uint32_t reason = read_u32();
      Card card = c_get_card(code);
      card.set_location(location);
      Card cnew = c_get_card(code);
      cnew.set_location(newloc);
      auto& pl = players_[card.controler_];
      auto& op = players_[1 - card.controler_];

      auto plspec = card.get_spec(false);
      auto opspec = card.get_spec(true);
      auto plnewspec = cnew.get_spec(false);
      auto opnewspec = cnew.get_spec(true);

      auto getspec = [&](auto& p) { return p.get() == pl.get() ? plspec : opspec; };
      auto getnewspec = [&](auto& p) {
        return p.get() == pl.get() ? plnewspec : opnewspec;
      };
      bool card_visible = true;
      if ((card.position_ & POS_FACEDOWN) && (cnew.position_ & POS_FACEDOWN)) {
        card_visible = false;
      }
      auto getvisiblename = [&](auto& p) {
        return card_visible ? card.name_ : "Face-down card";
      };

      if ((reason & REASON_DESTROY) && (card.location_ != cnew.location_)) {
        pl->notify(fmt::format("Card {} ({}) destroyed.", plspec, card.name_));
        op->notify(fmt::format("Card {} ({}) destroyed.", opspec, card.name_));
      } else if ((card.location_ == cnew.location_) &&
                 (card.location_ & LOCATION_ONFIELD)) {
        if (card.controler_ != cnew.controler_) {
          pl->notify(
              fmt::format("Your card {} ({}) changed controller to {} and is "
                          "now located at {}.",
                          plspec, card.name_, op->nickname_, plnewspec));
          op->notify(
              fmt::format("You now control {}'s card {} ({}) and it's located "
                          "at {}.",
                          pl->nickname_, opspec, card.name_, opnewspec));
        } else {
          pl->notify(fmt::format("Your card {} ({}) switched its zone to {}.",
                                 plspec, card.name_, plnewspec));
          op->notify(fmt::format("{}'s card {} ({}) switched its zone to {}.",
                                 pl->nickname_, opspec, card.name_, opnewspec));
        }
      } else if ((reason & REASON_DISCARD) &&
                 (card.location_ != cnew.location_)) {
        pl->notify(fmt::format("You discarded {} ({})", plspec, card.name_));
        op->notify(fmt::format("{} discarded {} ({})", pl->nickname_, opspec,
                               card.name_));
      } else if ((card.location_ == LOCATION_REMOVED) &&
                 (cnew.location_ & LOCATION_ONFIELD)) {
        pl->notify(
            fmt::format("Your banished card {} ({}) returns to the field at "
                        "{}.",
                        plspec, card.name_, plnewspec));
        op->notify(
            fmt::format("{}'s banished card {} ({}) returns to the field at "
                        "{}.",
                        pl->nickname_, opspec, card.name_, opnewspec));
      } else if ((card.location_ == LOCATION_GRAVE) &&
                 (cnew.location_ & LOCATION_ONFIELD)) {
        pl->notify(
            fmt::format("Your card {} ({}) returns from the graveyard to the "
                        "field at {}.",
                        plspec, card.name_, plnewspec));
        op->notify(
            fmt::format("{}'s card {} ({}) returns from the graveyard to the "
                        "field at {}.",
                        pl->nickname_, opspec, card.name_, opnewspec));
      } else if ((cnew.location_ == LOCATION_HAND) &&
                 (card.location_ != cnew.location_)) {
        pl->notify(
            fmt::format("Card {} ({}) returned to hand.", plspec, card.name_));
      } else if ((reason & (REASON_RELEASE | REASON_SUMMON)) &&
                 (card.location_ != cnew.location_)) {
        pl->notify(fmt::format("You tribute {} ({}).", plspec, card.name_));
        op->notify(fmt::format("{} tributes {} ({}).", pl->nickname_, opspec,
                               getvisiblename(op)));
      } else if ((card.location_ == (LOCATION_OVERLAY | LOCATION_MZONE)) &&
                 (cnew.location_ & LOCATION_GRAVE)) {
        pl->notify(fmt::format("You detached {}.", card.name_));
        op->notify(fmt::format("{} detached {}.", pl->nickname_, card.name_));
      } else if ((card.location_ != cnew.location_) &&
                 (cnew.location_ == LOCATION_GRAVE)) {
        pl->notify(fmt::format("Your card {} ({}) was sent to the graveyard.",
                               plspec, card.name_));
        op->notify(fmt::format("{}'s card {} ({}) was sent to the graveyard.",
                               pl->nickname_, opspec, card.name_));
      } else if ((card.location_ != cnew.location_) &&
                 (cnew.location_ == LOCATION_REMOVED)) {
        pl->notify(
            fmt::format("Your card {} ({}) was banished.", plspec, card.name_));
        op->notify(fmt::format("{}'s card {} ({}) was banished.", pl->nickname_,
                               opspec, getvisiblename(op)));
      } else if ((card.location_ != cnew.location_) &&
                 (cnew.location_ == LOCATION_DECK)) {
        pl->notify(fmt::format("Your card {} ({}) returned to your deck.",
                               plspec, card.name_));
        op->notify(fmt::format("{}'s card {} ({}) returned to their deck.",
                               pl->nickname_, opspec, getvisiblename(op)));
      } else if ((card.location_ != cnew.location_) &&
                 (cnew.location_ == LOCATION_EXTRA)) {
        pl->notify(fmt::format("Your card {} ({}) returned to your extra deck.",
                               plspec, card.name_));
        op->notify(
            fmt::format("{}'s card {} ({}) returned to their extra deck.",
                        pl->nickname_, opspec, getvisiblename(op)));
      } else if ((card.location_ == LOCATION_DECK) &&
                 (cnew.location_ == LOCATION_SZONE) &&
                 (cnew.position_ != POS_FACEDOWN)) {
        pl->notify(fmt::format("Activating {} ({})", plnewspec, card.name_));
        op->notify(fmt::format("{} activating {} ({})", pl->nickname_, opspec,
                               cnew.name_));
      }
    } else if (msg_ == MSG_SWAP) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      CardCode code1 = read_u32();
      uint32_t loc1 = read_u32();
      CardCode code2 = read_u32();
      uint32_t loc2 = read_u32();
      Card cards[2];
      cards[0] = c_get_card(code1);
      cards[1] = c_get_card(code2);
      cards[0].set_location(loc1);
      cards[1].set_location(loc2);

      for (PlayerId pl = 0; pl < 2; pl++) {
        for (int i = 0; i < 2; i++) {
          auto c = cards[i];
          auto spec = c.get_spec(pl);
          auto plname = players_[1 - c.controler_]->nickname_;
          players_[pl]->notify("Card " + c.name_ + " swapped control towards " +
                               plname + " and is now located at " + spec + ".");
        }
      }
    } else if (msg_ == MSG_SET) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      CardCode code = read_u32();
      uint32_t location = read_u32();
      Card card = c_get_card(code);
      card.set_location(location);
      auto c = card.controler_;
      auto& cpl = players_[c];
      auto& opl = players_[1 - c];
      cpl->notify(fmt::format("You set {} ({}) in {} position.", card.name_,
                              card.get_spec(c), card.get_position()));
      opl->notify(fmt::format("{} sets {} in {} position.", cpl->nickname_,
                              card.get_spec(PlayerId(1 - c)),
                              card.get_position()));
    } else if (msg_ == MSG_EQUIP) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto c = read_u8();
      auto loc = read_u8();
      auto seq = read_u8();
      auto pos = read_u8();
      Card card = get_card(c, loc, seq);
      c = read_u8();
      loc = read_u8();
      seq = read_u8();
      pos = read_u8();
      Card target = get_card(c, loc, seq);
      for (PlayerId pl = 0; pl < 2; pl++) {
        auto c = cardlist_info_for_player(card, pl);
        auto t = cardlist_info_for_player(target, pl);
        players_[pl]->notify(fmt::format("{} equipped to {}.", c, t));
      }
    } else if (msg_ == MSG_HINT) {
      auto hint_type = read_u8();
      auto player = read_u8();
      auto value = read_u32();

      if (hint_type == HINT_SELECTMSG && value == 501) {
        discard_hand_ = true;
      }
      // non-GUI don't need hint
      return;
      if (hint_type == HINT_SELECTMSG) {
        if (value > 2000) {
          CardCode code = value;
          players_[player]->notify(fmt::format("{} select {}",
                                               players_[player]->nickname_,
                                               c_get_card(code).name_));
        } else {
          players_[player]->notify(get_system_string(value));
        }
      } else if (hint_type == HINT_NUMBER) {
        players_[1 - player]->notify(
            fmt::format("Choice of player: {}", value));
      } else {
        fmt::println("Unknown hint type {} with value {}", hint_type, value);
      }
    } else if (msg_ == MSG_CARD_HINT) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      uint8_t player = read_u8();
      uint8_t loc = read_u8();
      uint8_t seq = read_u8();
      uint8_t pos = read_u8();
      uint8_t type = read_u8();
      uint32_t value = read_u32();
      if (type == CHINT_RACE) {
        Card card = get_card(player, loc, seq);
        if (card.code_ == 0) {
          return;
        }
        std::string races_str = "TODO";
        for (PlayerId pl = 0; pl < 2; pl++) {
          players_[pl]->notify(fmt::format("{} ({}) selected {}.",
                                           card.get_spec(pl), card.name_,
                                           races_str));
        }
      } else if (type == CHINT_ATTRIBUTE) {
        Card card = get_card(player, loc, seq);
        if (card.code_ == 0) {
          return;
        }
        std::string attributes_str = "TODO";
        for (PlayerId pl = 0; pl < 2; pl++) {
          players_[pl]->notify(fmt::format("{} ({}) selected {}.",
                                           card.get_spec(pl), card.name_,
                                           attributes_str));
        }
      } else {
        fmt::println("Unknown card hint type {} with value {}", type, value);
      }
    } else if (msg_ == MSG_POS_CHANGE) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      CardCode code = read_u32();
      Card card = c_get_card(code);
      card.set_location(read_u32());
      uint8_t prevpos = card.position_;
      card.position_ = read_u8();

      auto& pl = players_[card.controler_];
      auto& op = players_[1 - card.controler_];
      auto plspec = card.get_spec(false);
      auto opspec = card.get_spec(true);
      auto prevpos_str = position_to_string(prevpos);
      auto pos_str = position_to_string(card.position_);
      pl->notify("The position of card " + plspec + " (" + card.name_ +
                 ") changed from " + prevpos_str + " to " + pos_str + ".");
      op->notify("The position of card " + opspec + " (" + card.name_ +
                 ") changed from " + prevpos_str + " to " + pos_str + ".");
    } else if (msg_ == MSG_BECOME_TARGET) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto u = read_u8();
      uint32_t target = read_u32();
      uint8_t tc = target & 0xff;
      uint8_t tl = (target >> 8) & 0xff;
      uint8_t tseq = (target >> 16) & 0xff;
      Card card = get_card(tc, tl, tseq);
      auto name = players_[chaining_player_]->nickname_;
      for (PlayerId pl = 0; pl < 2; pl++) {
        auto spec = card.get_spec(pl);
        auto tcname = card.name_;
        if ((card.controler_ != pl) && (card.position_ & POS_FACEDOWN)) {
          tcname = position_to_string(card.position_) + " card";
        }
        players_[pl]->notify(name + " targets " + spec + " (" + tcname + ")");
      }
    } else if (msg_ == MSG_CONFIRM_DECKTOP) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto player = read_u8();
      auto size = read_u8();
      std::vector<Card> cards;
      for (int i = 0; i < size; ++i) {
        read_u32();
        auto c = read_u8();
        auto loc = read_u8();
        auto seq = read_u8();
        cards.push_back(get_card(c, loc, seq));
      }

      for (PlayerId pl = 0; pl < 2; pl++) {
        auto& p = players_[pl];
        if (pl == player) {
          p->notify(fmt::format("You reveal {} cards from your deck:", size));
        } else {
          p->notify(fmt::format("{} reveals {} cards from their deck:",
                                players_[player]->nickname_, size));
        }
        for (int i = 0; i < size; ++i) {
          p->notify(fmt::format("{}: {}", i + 1, cards[i].name_));
        }
      }
    } else if (msg_ == MSG_RANDOM_SELECTED) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto player = read_u8();
      auto count = read_u8();
      std::vector<Card> cards;

      for (int i = 0; i < count; ++i) {
        auto c = read_u8();
        auto loc = read_u8();
        if (loc & LOCATION_OVERLAY) {
          throw std::runtime_error("Overlay not supported for random selected");
        }
        auto seq = read_u8();
        auto pos = read_u8();
        cards.push_back(get_card(c, loc, seq));
      }

      for (PlayerId pl = 0; pl < 2; pl++) {
        auto& p = players_[pl];
        auto s = "card is";
        if (count > 1) {
          s = "cards are";
        }
        if (pl == player) {
          p->notify(fmt::format("Your {} {} randomly selected:", s, count));
        } else {
          p->notify(fmt::format("{}'s {} {} randomly selected:",
                                players_[player]->nickname_, s, count));
        }
        for (int i = 0; i < count; ++i) {
          p->notify(fmt::format("{}: {}", cards[i].get_spec(pl), cards[i].name_));
        }
      }
    } else if (msg_ == MSG_PLAYER_HINT) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      dp_ += 6;
      // TODO(3): implement output
    } else if (msg_ == MSG_CANCEL_TARGET) {
      const int payload_size = dl_ - dp_;
      if (payload_size != 8) {
        throw std::runtime_error(fmt::format(
            "Invalid cancel_target payload: {}", payload_size));
      }
      dp_ = dl_;
    } else if (msg_ == MSG_CARD_TARGET) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto c1 = read_u8();
      auto l1 = read_u8();
      auto s1 = read_u8();
      read_u8();
      auto c2 = read_u8();
      auto l2 = read_u8();
      auto s2 = read_u8();
      read_u8();

      Card card1 = get_card(c1, l1, s1);
      Card card2 = get_card(c2, l2, s2);
      for (PlayerId pl = 0; pl < 2; pl++) {
        auto& p = players_[pl];
        auto spec1 = card1.get_spec(pl);
        auto spec2 = card2.get_spec(pl);
        auto c1name = card1.name_;
        auto c2name = card2.name_;
        if ((card1.controler_ != pl) && (card1.position_ & POS_FACEDOWN)) {
          c1name = position_to_string(card1.position_) + " card";
        }
        if ((card2.controler_ != pl) && (card2.position_ & POS_FACEDOWN)) {
          c2name = position_to_string(card2.position_) + " card";
        }
        p->notify(fmt::format(" {} ({}) targets {} ({})", spec1, c1name, spec2, c2name));
      }
    } else if (msg_ == MSG_CONFIRM_CARDS) {
      auto player = read_u8();
      auto size = read_u8();
      std::vector<Card> cards;
      for (int i = 0; i < size; ++i) {
        read_u32();
        auto c = read_u8();
        auto loc = read_u8();
        auto seq = read_u8();
        if (verbose_) {
          cards.push_back(get_card(c, loc, seq));
        }
        revealed_.insert(ls_to_spec(loc, seq, 0, c == player));
      }
      if (!verbose_) {
        return;
      }

      auto& pl = players_[player];
      auto& op = players_[1 - player];

      op->notify(fmt::format("{} shows you {} cards.", pl->nickname_, size));
      for (int i = 0; i < size; ++i) {
        pl->notify(fmt::format("{}: {}", i + 1, cards[i].name_));
      }
    } else if (msg_ == MSG_MISSED_EFFECT) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      dp_ += 4;
      CardCode code = read_u32();
      Card card = c_get_card(code);
      for (PlayerId pl = 0; pl < 2; pl++) {
        auto spec = card.get_spec(pl);
        auto str = get_system_string(1622);
        std::string fmt_str = "[%ls]";
        str = str.replace(str.find(fmt_str), fmt_str.length(), card.name_);
        players_[pl]->notify(str);
      }
    } else if (msg_ == MSG_SORT_CARD) {
      // TODO(3): implement action
      if (!verbose_) {
        dp_ = dl_;
        resp_buf_[0] = 255;
        YGO_SetResponseb(pduel_, resp_buf_);
        return;
      }
      auto player = read_u8();
      auto size = read_u8();
      std::vector<Card> cards;
      for (int i = 0; i < size; ++i) {
        read_u32();
        auto c = read_u8();
        auto loc = read_u8();
        auto seq = read_u8();
        cards.push_back(get_card(c, loc, seq));
      }
      auto& pl = players_[player];
      pl->notify(
          "Sort " + std::to_string(size) +
          " cards by entering numbers separated by spaces (c = cancel):");
      for (int i = 0; i < size; ++i) {
        pl->notify(fmt::format("{}: {}", i + 1, cards[i].name_));
      }

      fmt::println("sort card action not implemented");
      resp_buf_[0] = 255;
      YGO_SetResponseb(pduel_, resp_buf_);

      // // generate all permutations
      // std::vector<int> perm(size);
      // std::iota(perm.begin(), perm.end(), 0);
      // std::vector<std::vector<int>> perms;
      // do {
      //   auto option = std::accumulate(perm.begin(), perm.end(),
      //   std::string(),
      //                                 [&](std::string &acc, int i) {
      //                                   return acc + std::to_string(i + 1) +
      //                                   " ";
      //                                 });
      //   options_.push_back(option);
      // } while (std::next_permutation(perm.begin(), perm.end()));
      // options_.push_back("c");
      // callback_ = [this](int idx) {
      //   const auto &option = options_[idx];
      //   if (option == "c") {
      //     resp_buf_[0] = 255;
      //     YGO_SetResponseb(pduel_, resp_buf_);
      //     return;
      //   }
      //   std::istringstream iss(option);
      //   int x;
      //   int i = 0;
      //   while (iss >> x) {
      //     resp_buf_[i] = uint8_t(x);
      //     i++;
      //   }
      //   YGO_SetResponseb(pduel_, resp_buf_);
      // };
    } else if (msg_ == MSG_ADD_COUNTER) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto ctype = read_u16();
      auto player = read_u8();
      auto loc = read_u8();
      auto seq = read_u8();
      auto count = read_u16();
      auto c = get_card(player, loc, seq);
      auto& pl = players_[player];
      PlayerId op_id = 1 - player;
      auto& op = players_[op_id];
      // TODO(3): counter type to string
      pl->notify(fmt::format("{} counter(s) of type {} placed on {} ().", count, "UNK", c.name_, c.get_spec(player)));
      op->notify(fmt::format("{} counter(s) of type {} placed on {} ().", count, "UNK", c.name_, c.get_spec(op_id)));
    } else if (msg_ == MSG_REMOVE_COUNTER) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto ctype = read_u16();
      auto player = read_u8();
      auto loc = read_u8();
      auto seq = read_u8();
      auto count = read_u16();
      auto c = get_card(player, loc, seq);
      auto& pl = players_[player];
      PlayerId op_id = 1 - player;
      auto& op = players_[op_id];
      pl->notify(fmt::format("{} counter(s) of type {} removed from {} ().", count, "UNK", c.name_, c.get_spec(player)));
      op->notify(fmt::format("{} counter(s) of type {} removed from {} ().", count, "UNK", c.name_, c.get_spec(op_id)));
    } else if (msg_ == MSG_ATTACK_DISABLED) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      for (PlayerId pl = 0; pl < 2; pl++) {
        players_[pl]->notify(get_system_string(1621));
      }
    } else if (msg_ == MSG_SHUFFLE_SET_CARD) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      // TODO(3): implement output
      dp_ = dl_;
    } else if (msg_ == MSG_SWAP_GRAVE_DECK) {
      if (dp_ >= dl_) {
        throw std::runtime_error("Truncated swap_grave_deck payload");
      }
      const auto player = read_u8();
      if (player > 1) {
        throw std::runtime_error(fmt::format(
            "Invalid swap_grave_deck player: {}", player));
      }
      const auto deck_count =
          YGO_QueryFieldCount(pduel_, player, LOCATION_DECK);
      const auto grave_count =
          YGO_QueryFieldCount(pduel_, player, LOCATION_GRAVE);
      if (deck_count < 0 || deck_count > 255 ||
          grave_count < 0 || grave_count > 255) {
        throw std::runtime_error(fmt::format(
            "Invalid swap_grave_deck zone counts: player={}, deck={}, grave={}",
            player, deck_count, grave_count));
      }
      record_protocol_event(
          player, msg_,
          {static_cast<uint8_t>(deck_count),
           static_cast<uint8_t>(grave_count)});
      if (verbose_) {
        auto& pl = players_[player];
        auto& op = players_[1 - player];
        pl->notify(fmt::format(
            "Your deck and graveyard were swapped (deck {}, grave {}).",
            deck_count, grave_count));
        op->notify(fmt::format(
            "{} swapped their deck and graveyard (deck {}, grave {}).",
            pl->nickname_, deck_count, grave_count));
      }
      return;
    } else if (msg_ == MSG_SHUFFLE_DECK) {
      if (dp_ >= dl_) {
        throw std::runtime_error("Truncated shuffle_deck payload");
      }
      const auto player = read_u8();
      if (player > 1) {
        throw std::runtime_error(fmt::format(
            "Invalid shuffle_deck player: {}", player));
      }
      if (!verbose_) {
        return;
      }
      auto& pl = players_[player];
      auto& op = players_[1 - player];
      pl->notify("You shuffled your deck.");
      op->notify(pl->nickname_ + " shuffled their deck.");
    } else if (msg_ == MSG_SHUFFLE_EXTRA) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto player = read_u8();
      auto count = read_u8();
      for (int i = 0; i < count; ++i) {
        read_u32();
      }
      auto& pl = players_[player];
      auto& op = players_[1 - player];
      pl->notify(fmt::format("You shuffled your extra deck ({}).", count));
      op->notify(fmt::format("{} shuffled their extra deck ({}).", pl->nickname_, count));
    } else if (msg_ == MSG_SHUFFLE_HAND) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }

      auto player = read_u8();
      dp_ = dl_;

      auto& pl = players_[player];
      auto& op = players_[1 - player];
      pl->notify("You shuffled your hand.");
      op->notify(pl->nickname_ + " shuffled their hand.");
    } else if (msg_ == MSG_SUMMONED) {
      dp_ = dl_;
    } else if (msg_ == MSG_SUMMONING) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      CardCode code = read_u32();
      Card card = c_get_card(code);
      card.set_location(read_u32());
      const auto &nickname = players_[card.controler_]->nickname_;
      for (auto& pl : players_) {
        pl->notify(nickname + " summoning " + card.name_ + " (" +
                   std::to_string(card.attack_) + "/" +
                   std::to_string(card.defense_) + ") in " +
                   card.get_position() + " position.");
      }
    } else if (msg_ == MSG_SPSUMMONED) {
      dp_ = dl_;
    } else if (msg_ == MSG_FLIPSUMMONED) {
      dp_ = dl_;
    } else if (msg_ == MSG_FLIPSUMMONING) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }

      auto code = read_u32();
      auto location = read_u32();
      Card card = c_get_card(code);
      card.set_location(location);

      auto& cpl = players_[card.controler_];
      for (PlayerId pl = 0; pl < 2; pl++) {
        auto spec = card.get_spec(pl);
        players_[1 - pl]->notify(cpl->nickname_ + " flip summons " + spec +
                                 " (" + card.name_ + ")");
      }
    } else if (msg_ == MSG_SPSUMMONING) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      CardCode code = read_u32();
      Card card = c_get_card(code);
      card.set_location(read_u32());
      const auto &nickname = players_[card.controler_]->nickname_;
      for (PlayerId p = 0; p < 2; p++) {
        auto& pl = players_[p];
        auto pos = card.get_position();
        auto atk = std::to_string(card.attack_);
        auto def = std::to_string(card.defense_);
        std::string name = p == card.controler_ ? "You" : nickname;
        if (card.type_ & TYPE_LINK) {
          pl->notify(name + " special summoning " + card.name_ + " (" +
                     atk + ") in " + pos + " position.");
        } else {
          pl->notify(name + " special summoning " + card.name_ + " (" +
                     atk + "/" + def + ") in " + pos + " position.");
        }
      }
    } else if (msg_ == MSG_CHAIN_NEGATED) {
      dp_ = dl_;
    } else if (msg_ == MSG_CHAIN_DISABLED) {
      dp_ = dl_;
    } else if (msg_ == MSG_CHAIN_SOLVED) {
      dp_ = dl_;
      revealed_.clear();
    } else if (msg_ == MSG_CHAIN_SOLVING) {
      dp_ = dl_;
    } else if (msg_ == MSG_CHAINED) {
      dp_ = dl_;
    } else if (msg_ == MSG_CHAIN_END) {
      dp_ = dl_;
    } else if (msg_ == MSG_CHAINING) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      CardCode code = read_u32();
      Card card = c_get_card(code);
      card.set_location(read_u32());
      auto tc = read_u8();
      auto tl = read_u8();
      auto ts = read_u8();
      uint32_t desc = read_u32();
      auto cs = read_u8();
      auto c = card.controler_;
      PlayerId o = 1 - c;
      chaining_player_ = c;
      players_[c]->notify("Activating " + card.get_spec(c) + " (" + card.name_ +
                          ")");
      players_[o]->notify(players_[c]->nickname_ + " activating " +
                          card.get_spec(o) + " (" + card.name_ + ")");
    } else if (msg_ == MSG_DAMAGE) {
      auto player = read_u8();
      auto amount = read_u32();
      _damage(player, amount);
    } else if (msg_ == MSG_RECOVER) {
      auto player = read_u8();
      auto amount = read_u32();
      _recover(player, amount);
    } else if (msg_ == MSG_LPUPDATE) {
      auto player = read_u8();
      auto lp = read_u32();
      if (lp >= lp_[player]) {
        _recover(player, lp - lp_[player]);
      } else {
        _damage(player, lp_[player] - lp);
      }
    } else if (msg_ == MSG_PAY_LPCOST) {
      auto player = read_u8();
      auto cost = read_u32();
      lp_[player] -= cost;
      if (!verbose_) {
        return;
      }
      auto& pl = players_[player];
      pl->notify("You pay " + std::to_string(cost) + " LP. Your LP is now " +
                 std::to_string(lp_[player]) + ".");
      players_[1 - player]->notify(
          pl->nickname_ + " pays " + std::to_string(cost) + " LP. " +
          pl->nickname_ + "'s LP is now " + std::to_string(lp_[player]) + ".");
    } else if (msg_ == MSG_ATTACK) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto attacker = read_u32();
      PlayerId ac = attacker & 0xff;
      auto aloc = (attacker >> 8) & 0xff;
      auto aseq = (attacker >> 16) & 0xff;
      auto apos = (attacker >> 24) & 0xff;
      auto target = read_u32();
      PlayerId tc = target & 0xff;
      auto tloc = (target >> 8) & 0xff;
      auto tseq = (target >> 16) & 0xff;
      auto tpos = (target >> 24) & 0xff;

      if ((ac == 0) && (aloc == 0) && (aseq == 0) && (apos == 0)) {
        return;
      }

      Card acard = get_card(ac, aloc, aseq);
      auto name = players_[ac]->nickname_;
      if ((tc == 0) && (tloc == 0) && (tseq == 0) && (tpos == 0)) {
        for (PlayerId i = 0; i < 2; i++) {
          players_[i]->notify(name + " prepares to attack with " +
                              acard.get_spec(i) + " (" + acard.name_ + ")");
        }
        return;
      }

      Card tcard = get_card(tc, tloc, tseq);
      for (PlayerId i = 0; i < 2; i++) {
        auto aspec = acard.get_spec(i);
        auto tspec = tcard.get_spec(i);
        auto tcname = tcard.name_;
        if ((tcard.controler_ != i) && (tcard.position_ & POS_FACEDOWN)) {
          tcname = tcard.get_position() + " card";
        }
        players_[i]->notify(name + " prepares to attack " + tspec + " (" +
                            tcname + ") with " + aspec + " (" + acard.name_ +
                            ")");
      }
    } else if (msg_ == MSG_DAMAGE_STEP_START) {
      if (!verbose_) {
        return;
      }
      for (int i = 0; i < 2; i++) {
        players_[i]->notify("begin damage");
      }
    } else if (msg_ == MSG_DAMAGE_STEP_END) {
      if (!verbose_) {
        return;
      }
      for (int i = 0; i < 2; i++) {
        players_[i]->notify("end damage");
      }
    } else if (msg_ == MSG_BATTLE) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto attacker = read_u32();
      auto aa = read_u32();
      auto ad = read_u32();
      auto bd0 = read_u8();
      auto target = read_u32();
      auto da = read_u32();
      auto dd = read_u32();
      auto bd1 = read_u8();

      auto ac = attacker & 0xff;
      auto aloc = (attacker >> 8) & 0xff;
      auto aseq = (attacker >> 16) & 0xff;

      auto tc = target & 0xff;
      auto tloc = (target >> 8) & 0xff;
      auto tseq = (target >> 16) & 0xff;
      auto tpos = (target >> 24) & 0xff;

      Card acard = get_card(ac, aloc, aseq);
      Card tcard;
      if (tloc != 0) {
        tcard = get_card(tc, tloc, tseq);
      }
      for (int i = 0; i < 2; i++) {
        auto& pl = players_[i];
        std::string attacker_points;
        if (acard.type_ & TYPE_LINK) {
          attacker_points = std::to_string(aa);
        } else {
          attacker_points = std::to_string(aa) + "/" + std::to_string(ad);
        }
        if (tloc != 0) {
          std::string defender_points;
          if (tcard.type_ & TYPE_LINK) {
            defender_points = std::to_string(da);
          } else {
            defender_points = std::to_string(da) + "/" + std::to_string(dd);
          }
          pl->notify(acard.name_ + "(" + attacker_points + ")" + " attacks " +
                     tcard.name_ + " (" + defender_points + ")");
        } else {
          pl->notify(acard.name_ + "(" + attacker_points + ")" + " attacks");
        }
      }
    } else if (msg_ == MSG_WIN) {
      auto player = read_u8();
      auto reason = read_u8();
      auto& winner = players_[player];
      auto& loser = players_[1 - player];

      _duel_end(player, reason);

      auto l_reason = reason_to_string(reason);
      if (verbose_) {
        winner->notify("You won (" + l_reason + ").");
        loser->notify("You lost (" + l_reason + ").");
      }
    } else if (msg_ == MSG_RETRY) {
      throw std::runtime_error("Retry");
    } else if (msg_ == MSG_SELECT_BATTLECMD) {
      auto player = read_u8();
      auto activatable = read_cardlist_spec(player, true);
      auto attackable = read_cardlist_spec(player, true, true);
      bool to_m2 = read_u8();
      bool to_ep = read_u8();

      auto& pl = players_[player];
      if (verbose_) {
        pl->notify("Battle menu:");
      }
      for (const auto [code_t, spec, desc] : activatable) {
        CardCode code = code_t;
        if(code & 0x80000000) {
          code &= 0x7fffffff;
        }
        auto [code_d, eff_idx] = unpack_desc(code, desc);
        if (desc == 0) {
          code_d = code;
        }
        auto la = LegalAction::activate_spec(eff_idx, spec);
        if (code_d != 0) {
          la.cid_ = c_get_card_id(code_d);
        }
        legal_actions_.push_back(la);
        if (verbose_) {
          auto c = c_get_card(code);
          int cmd_idx = legal_actions_.size();
          std::string s = fmt::format(
            "{}: activate {}({}) [{}/{}] ({})",
            cmd_idx, c.name_, spec, c.attack_, c.defense_, c.get_effect_description(code_d, eff_idx));
        }
      }
      for (const auto [code, spec, data] : attackable) {
        bool direct_attackable = data & 0x1;
        auto act = direct_attackable ? ActionAct::DirectAttack : ActionAct::Attack;

        legal_actions_.push_back(
          LegalAction::act_spec(act, spec));
        if (verbose_) {
          auto [controller, loc, seq, pos] = spec_to_ls(player, spec);
          auto c = get_card(controller, loc, seq);
          int cmd_idx = legal_actions_.size();
          auto attack_str = direct_attackable ? "direct attack" : "attack";
          std::string s = fmt::format(
            "{}: {} {}({}) ", cmd_idx, attack_str, c.name_, spec);
          if (c.type_ & TYPE_LINK) {
            s += fmt::format("[{}]", c.attack_);
          } else {
            s += fmt::format("[{}/{}]", c.attack_, c.defense_);
          }
          pl->notify(s);
        }
      }
      if (to_m2) {
        legal_actions_.push_back(
          LegalAction::phase(ActionPhase::Main2));
        int cmd_idx = legal_actions_.size();
        if (verbose_) {
          pl->notify(fmt::format("{}: Main phase 2.", cmd_idx));
        }
      }
      if (to_ep) {
        if (!to_m2) {
          legal_actions_.push_back(
            LegalAction::phase(ActionPhase::End));
          int cmd_idx = legal_actions_.size();
          if (verbose_) {
            pl->notify(fmt::format("{}: End phase.", cmd_idx));
          }
        }
      }
      int n_activatables = activatable.size();
      int n_attackables = attackable.size();
      to_play_ = player;
      callback_ = [this, n_activatables, n_attackables, to_ep, to_m2](int idx) {
        const auto &la = legal_actions_[idx];
        if (idx < n_activatables) {
          YGO_SetResponsei(pduel_, idx << 16);
        } else if (idx < (n_activatables + n_attackables)) {
          idx = idx - n_activatables;
          YGO_SetResponsei(pduel_, (idx << 16) + 1);
        } else if ((la.phase_ == ActionPhase::End) && to_ep) {
          YGO_SetResponsei(pduel_, 3);
        } else if ((la.phase_ == ActionPhase::Main2) && to_m2) {
          YGO_SetResponsei(pduel_, 2);
        } else {
          throw std::runtime_error("Invalid option");
        }
      };
    } else if (msg_ == MSG_SELECT_UNSELECT_CARD) {
      // TODO: add feature of selected cards (also for multi select)
      auto player = read_u8();
      bool finishable = read_u8();
      bool cancelable = read_u8();
      auto min = read_u8();
      auto max = read_u8();
      auto select_size = read_u8();

      std::vector<std::string> select_specs;
      select_specs.reserve(select_size);
      if (verbose_) {
        auto& pl = players_[player];
        pl->notify("Select " + std::to_string(min) + " to " +
                   std::to_string(max) + " cards:");
        for (int i = 0; i < select_size; ++i) {
          auto code = read_u32();
          auto loc = read_u32();
          Card card = c_get_card(code);
          card.set_location(loc);
          auto spec = card.get_spec(player);
          select_specs.push_back(spec);
          auto s = fmt::format("{}: {}({})", i + 1, card.name_, spec);
          pl->notify(s);
        }
      } else {
        for (int i = 0; i < select_size; ++i) {
          dp_ += 4;
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto pos = read_u8();
          auto spec = ls_to_spec(loc, seq, pos, controller != player);
          select_specs.push_back(spec);
        }
      }

      auto unselect_size = read_u8();

      // unselect not allowed (no regrets)
      dp_ += 8 * unselect_size;

      for (int j = 0; j < select_specs.size(); ++j) {
        legal_actions_.push_back(LegalAction::from_spec(select_specs[j]));
      }

      if (finishable) {
        legal_actions_.push_back(LegalAction::finish());
      }

      // cancelable and finishable not needed

      to_play_ = player;
      callback_ = [this](int idx) {
        if (legal_actions_[idx].finish_) {
          YGO_SetResponsei(pduel_, -1);
        } else {
          resp_buf_[0] = 1;
          resp_buf_[1] = idx;
          YGO_SetResponseb(pduel_, resp_buf_);
        }
      };

    } else if (msg_ == MSG_SELECT_CARD) {
      auto player = read_u8();
      bool cancelable = read_u8();
      auto min = read_u8();
      auto max = read_u8();
      auto size = read_u8();

      if (min == 0) {
        throw std::runtime_error("Min == 0 not implemented for select card");
      }

      std::vector<std::string> specs;
      specs.reserve(size);
      if (verbose_) {
        std::vector<Card> cards;
        for (int i = 0; i < size; ++i) {
          auto code = read_u32();
          auto loc = read_u32();
          Card card = c_get_card(code);
          card.set_location(loc);
          cards.push_back(card);
        }
        auto& pl = players_[player];
        pl->notify("Select " + std::to_string(min) + " to " +
                   std::to_string(max) + " cards separated by spaces:");
        for (const auto &card : cards) {
          auto spec = card.get_spec(player);
          specs.push_back(spec);
          int i = specs.size();
          if (card.controler_ != player && card.position_ & POS_FACEDOWN) {
            pl->notify(
              fmt::format("{}: {} card ({})", i, card.get_position(), spec));
          } else {
            pl->notify(
              fmt::format("{}: {} ({})", i, card.name_, spec));
          }
        }
      } else {
        for (int i = 0; i < size; ++i) {
          dp_ += 4;
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto pos = read_u8();
          auto spec = ls_to_spec(loc, seq, pos, controller != player);
          specs.push_back(spec);
        }
      }

      if (discard_hand_) {
        discard_hand_ = false;
        if (current_phase_ == PHASE_END) {
          // random discard
          std::vector<int> comb(size);
          std::iota(comb.begin(), comb.end(), 0);
          std::shuffle(comb.begin(), comb.end(), gen_);
          resp_buf_[0] = min;
          for (int i = 0; i < min; ++i) {
            resp_buf_[i + 1] = comb[i];
          }
          YGO_SetResponseb(pduel_, resp_buf_);
          return;
        }
      }

      // TODO(1): use this when added to history actions
      // if ((min == max) && (max == specs.size())) {
      //   resp_buf_[0] = specs.size();
      //   for (int i = 0; i < specs.size(); ++i) {
      //     resp_buf_[i + 1] = i;
      //   }
      //   YGO_SetResponseb(pduel_, resp_buf_);
      //   return;
      // }

      init_multi_select(min, max, 0, specs);

      to_play_ = player;
      callback_ = [this](int idx) {
        _callback_multi_select(idx, ms_max_ == 1);
      };
    } else if (msg_ == MSG_SELECT_TRIBUTE) {
      auto player = read_u8();
      bool cancelable = read_u8();
      auto min = read_u8();
      auto max = read_u8();
      auto size = read_u8();

      std::vector<int> release_params;
      release_params.reserve(size);
      std::vector<std::string> specs;
      specs.reserve(size);
      if (verbose_) {
        std::vector<Card> cards;
        for (int i = 0; i < size; ++i) {
          auto code = read_u32();
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto release_param = read_u8();
          Card card = get_card(controller, loc, seq);
          cards.push_back(card);
          release_params.push_back(release_param);
        }
        auto& pl = players_[player];
        pl->notify("Select " + std::to_string(min) + " to " +
                   std::to_string(max) +
                   " cards to tribute separated by spaces:");
        for (const auto &card : cards) {
          auto spec = card.get_spec(player);
          specs.push_back(spec);
          pl->notify(
            fmt::format("{}: {} ({})", specs.size(), card.name_, spec));
        }
      } else {
        for (int i = 0; i < size; ++i) {
          dp_ += 4;
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto release_param = read_u8();

          auto spec = ls_to_spec(loc, seq, 0, controller != player);
          specs.push_back(spec);

          release_params.push_back(release_param);
        }
      }

      const auto combs = tribute_combinations(release_params, min, max);
      if (combs.empty()) {
        throw std::runtime_error(fmt::format(
            "No valid select tribute combination: min={}, max={}, "
            "release_params={}",
            min, max, release_params));
      }

      init_multi_select(min, max, 0, specs, 1, combs);

      to_play_ = player;
      callback_ = [this](int idx) {
        _callback_multi_select_2(idx);
      };
    } else if (msg_ == MSG_SELECT_SUM) {
      auto mode = read_u8();
      auto player = read_u8();
      auto val = read_u32();
      int _min = read_u8();
      int _max = read_u8();
      auto must_select_size = read_u8();

      if (mode > 1) {
        throw std::runtime_error("mode: " + std::to_string(mode) +
                                 " not implemented for MSG_SELECT_SUM");
      }
      if (must_select_size > 2) {
        throw std::runtime_error(
            " must select size: " + std::to_string(must_select_size) +
            " not implemented for MSG_SELECT_SUM");
      }

      std::vector<uint32_t> must_select_params;
      std::vector<uint32_t> select_params;
      std::vector<std::string> select_specs;
      must_select_params.reserve(must_select_size);

      if (verbose_) {
        std::vector<Card> must_select;
        must_select.reserve(must_select_size);
        for (int i = 0; i < must_select_size; ++i) {
          auto code = read_u32();
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto param = read_u32();
          Card card = get_card(controller, loc, seq);
          must_select.push_back(card);
          must_select_params.push_back(param);
        }
        auto& pl = players_[player];
        pl->notify("Select cards for a target value of " +
                   std::to_string(val) + ", separated by spaces.");
        for (const auto &card : must_select) {
          auto spec = card.get_spec(player);
          pl->notify(card.name_ + " (" + spec +
                     ") must be selected, automatically selected.");
        }
      } else {
        for (int i = 0; i < must_select_size; ++i) {
          dp_ += 4;
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto param = read_u32();

          auto spec = ls_to_spec(loc, seq, 0, controller != player);
          must_select_params.push_back(param);
        }
      }

      uint8_t select_size = read_u8();
      select_params.reserve(select_size);
      select_specs.reserve(select_size);

      if (verbose_) {
        std::vector<Card> select;
        select.reserve(select_size);
        for (int i = 0; i < select_size; ++i) {
          auto code = read_u32();
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto param = read_u32();
          Card card = get_card(controller, loc, seq);
          select.push_back(card);
          select_params.push_back(param);
        }
        auto& pl = players_[player];
        for (const auto &card : select) {
          auto spec = card.get_spec(player);
          select_specs.push_back(spec);
          pl->notify(
            fmt::format("{}: {} ({})", select_specs.size(), card.name_, spec));
        }
      } else {
        for (int i = 0; i < select_size; ++i) {
          dp_ += 4;
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto param = read_u32();

          auto spec = ls_to_spec(loc, seq, 0, controller != player);
          select_specs.push_back(spec);
          select_params.push_back(param);
        }
      }

      std::vector<std::vector<int>> combs =
          combinations_with_sum_params(
              must_select_params, select_params, val, _min, _max,
              mode == 1);
      if (combs.empty()) {
        throw std::runtime_error(fmt::format(
            "No valid MSG_SELECT_SUM combination: mode={}, target={}, "
            "min={}, max={}, must={}, candidates={}",
            mode, val, _min, _max, must_select_size, select_size));
      }
      if (combs.front().empty()) {
        resp_buf_[0] = must_select_size;
        for (int i = 0; i < must_select_size; ++i) {
          resp_buf_[i + 1] = 0;
        }
        YGO_SetResponseb(pduel_, resp_buf_);
        return;
      }

      int selection_min = _min;
      int selection_max = _max;
      if (mode == 1) {
        selection_min = select_size;
        selection_max = 0;
        for (const auto &comb : combs) {
          selection_min = std::min(
              selection_min, static_cast<int>(comb.size()));
          selection_max = std::max(
              selection_max, static_cast<int>(comb.size()));
        }
      }
      init_multi_select(
          selection_min, selection_max, must_select_size, select_specs,
          mode + 1, combs);

      to_play_ = player;
      callback_ = [this](int idx) {
        _callback_multi_select_2(idx);
      };

    } else if (msg_ == MSG_SELECT_CHAIN) {
      auto player = read_u8();
      auto size = read_u8();
      auto spe_count = read_u8();
      bool forced = read_u8();
      dp_ += 8;
      // auto hint_timing = read_u32();
      // auto other_timing = read_u32();

      std::vector<CardCode> codes;
      std::vector<uint32_t> descs;
      std::vector<std::string> specs;
      for (int i = 0; i < size; ++i) {
        auto flag = read_u8();
        CardCode code = read_u32();
        codes.push_back(code);
        PlayerId c = read_u8();
        uint8_t loc = read_u8();
        uint8_t seq = read_u8();
        uint8_t pos = read_u8();
        specs.push_back(ls_to_spec(loc, seq, pos, c != player));
        uint32_t desc = read_u32();
        descs.push_back(desc);
      }

      if ((size == 0) && (spe_count == 0)) {
        // non-GUI don't need this
        // if (verbose_) {
        //   fmt::println("keep processing");
        // }
        YGO_SetResponsei(pduel_, -1);
        return;
      }

      auto& pl = players_[player];
      auto& op = players_[1 - player];
      chaining_player_ = player;
      if (!op->seen_waiting_) {
        if (verbose_) {
          op->notify("Waiting for opponent.");
        }
        op->seen_waiting_ = true;
      }

      if (verbose_) {
        pl->notify("Select chain:");
      }

      for (int i = 0; i < size; i++) {
        CardCode code = codes[i];
        uint32_t desc = descs[i];
        auto spec = specs[i];
        auto [code_d, eff_idx] = unpack_desc(code, desc);
        if (desc == 0) {
          code_d = code;
        }
        auto la = LegalAction::activate_spec(eff_idx, spec);
        if (code_d != 0) {
          la.cid_ = c_get_card_id(code_d);
        }
        legal_actions_.push_back(la);
        if (verbose_) {
          auto c = c_get_card(code);
          std::string s = fmt::format(
            "{}: {}({}) ({})",
            i + 1, c.name_, spec, c.get_effect_description(code_d, eff_idx));
          pl->notify(s);
        }
      }

      if (!forced) {
        legal_actions_.push_back(LegalAction::cancel());
        if (verbose_) {
          pl->notify(fmt::format("{}: cancel", size + 1));
        }
      }
      to_play_ = player;
      callback_ = [this, forced](int idx) {
        const auto &action = legal_actions_[idx];
        if (action.act_ == ActionAct::Cancel) {
          if (forced) {
            fmt::print("cancel not allowed in forced chain\n");
            YGO_SetResponsei(pduel_, 0);
            return;
          }
          YGO_SetResponsei(pduel_, -1);
          return;
        }
        YGO_SetResponsei(pduel_, idx);
      };
    } else if (msg_ == MSG_SELECT_YESNO) {
      auto player = read_u8();
      auto desc = read_u32();
      auto [code, eff_idx] = unpack_desc(0, desc);
      if (desc == 0) {
        show_buffer();
        auto s = fmt::format("Unknown desc {} in select_yesno", desc);
        throw std::runtime_error(s);
      }
      auto la = LegalAction::activate_spec(eff_idx, "");
      if (code != 0) {
        la.cid_ = c_get_card_id(code);
      }
      legal_actions_.push_back(la);
      if (verbose_) {
        auto& pl = players_[player];
        std::string s;
        if (code == 0) {
          s = get_system_string(eff_idx);
        } else {
          Card c = c_get_card(code);
          int cmd_idx = legal_actions_.size();
          eff_idx -= CARD_EFFECT_OFFSET;
          if (eff_idx >= c.strings_.size()) {
            throw std::runtime_error(
              fmt::format("Unknown effect {} of {}", eff_idx, c.name_));
          }
          auto str = c.strings_[eff_idx];
          if (str.empty()) {
            str = "effect " + std::to_string(eff_idx);
          }
          s = fmt::format("{} ({})", c.name_, str);
        }
        pl->notify("1: " + s);
        pl->notify("2: No");
      }
      // TODO: maybe add card id to cancel
      legal_actions_.push_back(LegalAction::cancel());
      to_play_ = player;
      callback_ = [this](int idx) {
        if (idx == 0) {
          YGO_SetResponsei(pduel_, 1);
        } else if (idx == 1) {
          YGO_SetResponsei(pduel_, 0);
        }
      };
    } else if (msg_ == MSG_SELECT_EFFECTYN) {
      auto player = read_u8();

      CardCode code = read_u32();
      auto ct = read_u8();
      auto loc = read_u8();
      auto seq = read_u8();
      auto pos = read_u8();
      auto desc = read_u32();
      std::string spec = ls_to_spec(loc, seq, pos, ct != player);
      auto [code_d, eff_idx] = unpack_desc(code, desc);
      if (desc == 0) {
        code_d = code;
      }
      auto la = LegalAction::activate_spec(eff_idx, spec);
      if (code_d != 0) {
        la.cid_ = c_get_card_id(code_d);
      }
      legal_actions_.push_back(la);

      if (verbose_) {
        Card c = c_get_card(code);
        auto& pl = players_[player];
        auto name = c.name_;
        std::string s;
        if (code_d == 0) {
          s = get_system_string(desc);
          std::string fmt_str = "[%ls]";
          auto pos = find_substrs(s, fmt_str);
          if (pos.size() == 0) {
            // nothing to replace
          } else if (pos.size() == 1) {
            auto p = pos[0];
            s = s.substr(0, p) + name + s.substr(p + fmt_str.size());
          } else if (pos.size() == 2) {
            auto p1 = pos[0];
            auto p2 = pos[1];
            s = s.substr(0, p1) + spec +
                s.substr(p1 + fmt_str.size(), p2 - p1 - fmt_str.size()) + name +
                s.substr(p2 + fmt_str.size());
          } else {
            throw std::runtime_error("Unknown effectyn desc " +
                                     std::to_string(desc) + " of " + name);
          }
        } else {
          s = fmt::format(
            "{}({}) ({})", c.name_, spec, c.get_effect_description(code_d, eff_idx));
        }
        pl->notify("1: " + s);
        pl->notify("2: No");
      }

      // TODO: maybe add card info to cancel
      legal_actions_.push_back(LegalAction::cancel());
      to_play_ = player;
      callback_ = [this](int idx) {
        if (idx == 0) {
          YGO_SetResponsei(pduel_, 1);
        } else if (idx == 1) {
          YGO_SetResponsei(pduel_, 0);
        }
      };
    } else if (msg_ == MSG_SELECT_OPTION) {
      auto player = read_u8();
      auto size = read_u8();
      if (verbose_) {
        players_[player]->notify("Select an option:");
      }
      for (int i = 0; i < size; ++i) {
        auto desc = read_u32();
        auto [code, eff_idx] = unpack_desc(0, desc);
        if (desc == 0) {
          show_buffer();
          auto s = fmt::format("Unknown desc {} in select_option", desc);
          throw std::runtime_error(s);
        }
        auto la = LegalAction::activate_spec(eff_idx, "");
        if (code != 0) {
          la.cid_ = c_get_card_id(code);
        }
        legal_actions_.push_back(la);
        if (verbose_) {
          std::string s;
          if (code == 0) {
            s = get_system_string(eff_idx);
          } else {
            Card c = c_get_card(code);
            int cmd_idx = legal_actions_.size();
            eff_idx -= CARD_EFFECT_OFFSET;
            if (eff_idx >= c.strings_.size()) {
              throw std::runtime_error(
                fmt::format("Unknown effect {} of {}", eff_idx, c.name_));
            }
            auto str = c.strings_[eff_idx];
            if (str.empty()) {
              str = "effect " + std::to_string(eff_idx);
            }
            s = fmt::format("{} ({})", c.name_, str);
          }
          players_[player]->notify(std::to_string(i + 1) + ": " + s);
        }
      }

      to_play_ = player;
      callback_ = [this](int idx) {
        YGO_SetResponsei(pduel_, idx);
      };
    } else if (msg_ == MSG_SELECT_IDLECMD) {
      int32_t player = read_u8();
      auto summonable_ = read_cardlist_spec(player);
      auto spsummon_ = read_cardlist_spec(player);
      auto repos_ = read_cardlist_spec(player);
      auto idle_mset_ = read_cardlist_spec(player);
      auto idle_set_ = read_cardlist_spec(player);
      auto idle_activate_ = read_cardlist_spec(player, true);
      bool to_bp_ = read_u8();
      bool to_ep_ = read_u8();
      read_u8(); // can_shuffle

      int offset = 0;

      auto& pl = players_[player];
      if (verbose_) {
        pl->notify("Select a card and action to perform.");
      }
      for (const auto &[code, spec, data] : summonable_) {
        legal_actions_.push_back(LegalAction::act_spec(ActionAct::Summon, spec));
        if (verbose_) {
          const auto &name = c_get_card(code).name_;
          int cmd_idx = legal_actions_.size();
          pl->notify(fmt::format(
            "{}: Summon {} in face-up attack position", cmd_idx, name));
        }
      }
      offset += summonable_.size();
      int spsummon_offset = offset;
      for (const auto &[code, spec, data] : spsummon_) {
        legal_actions_.push_back(LegalAction::act_spec(ActionAct::SpSummon, spec));
        if (verbose_) {
          const auto &name = c_get_card(code).name_;
          int cmd_idx = legal_actions_.size();
          pl->notify(fmt::format(
            "{}: Special summon {}", cmd_idx, name));
        }
      }
      offset += spsummon_.size();
      int repos_offset = offset;
      for (const auto &[code, spec, data] : repos_) {
        legal_actions_.push_back(LegalAction::act_spec(ActionAct::Repo, spec));
        if (verbose_) {
          const auto &name = c_get_card(code).name_;
          int cmd_idx = legal_actions_.size();
          pl->notify(fmt::format(
            "{}: Change position of {}", cmd_idx, name));
        }
      }
      offset += repos_.size();
      int mset_offset = offset;
      for (const auto &[code, spec, data] : idle_mset_) {
        legal_actions_.push_back(LegalAction::act_spec(ActionAct::MSet, spec));
        if (verbose_) {
          const auto &name = c_get_card(code).name_;
          int cmd_idx = legal_actions_.size();
          pl->notify(fmt::format(
            "{}: Summon {} in face-down defense position", cmd_idx, name));
        }
      }
      offset += idle_mset_.size();
      int set_offset = offset;
      for (const auto &[code, spec, data] : idle_set_) {
        legal_actions_.push_back(LegalAction::act_spec(ActionAct::Set, spec));
        if (verbose_) {
          const auto &name = c_get_card(code).name_;
          int cmd_idx = legal_actions_.size();
          pl->notify(fmt::format(
            "{}: Set {}", cmd_idx, name));
        }
      }
      offset += idle_set_.size();
      int activate_offset = offset;
      for (const auto &[code_t, spec, desc] : idle_activate_) {
        CardCode code = code_t;
        if(code & 0x80000000) {
          code &= 0x7fffffff;
        }
        auto [code_d, eff_idx] = unpack_desc(code, desc);
        if (desc == 0) {
          code_d = code;
        }
        auto la = LegalAction::activate_spec(eff_idx, spec);
        if (code_d != 0) {
          la.cid_ = c_get_card_id(code_d);
        }
        legal_actions_.push_back(la);
        if (verbose_) {
          auto c = c_get_card(code);
          int cmd_idx = legal_actions_.size();
          std::string s = fmt::format(
            "{}: Activate {}({}) ({})",
            cmd_idx, c.name_, spec, c.get_effect_description(code_d, eff_idx));
          pl->notify(s);
        }
      }

      if (to_bp_) {
        legal_actions_.push_back(LegalAction::phase(ActionPhase::Battle));
        if (verbose_) {
          int cmd_idx = legal_actions_.size();
          pl->notify(fmt::format("{}: Enter the battle phase.", cmd_idx));
        }
      }
      if (to_ep_) {
        if (!to_bp_) {
          legal_actions_.push_back(LegalAction::phase(ActionPhase::End));
          if (verbose_) {
            int cmd_idx = legal_actions_.size();
            pl->notify(fmt::format("{}: End phase.", cmd_idx));
          }
        }
      }

      to_play_ = player;
      callback_ = [this, spsummon_offset, repos_offset, mset_offset, set_offset,
                   activate_offset](int idx) {
        const auto &action = legal_actions_[idx];
        if (action.phase_ == ActionPhase::Battle) {
          YGO_SetResponsei(pduel_, 6);
        } else if (action.phase_ == ActionPhase::End) {
          YGO_SetResponsei(pduel_, 7);
        } else {
          auto act = action.act_;
          if (act == ActionAct::Summon) {
            uint32_t idx_ = idx;
            YGO_SetResponsei(pduel_, idx_ << 16);
          } else if (act == ActionAct::SpSummon) {
            uint32_t idx_ = idx - spsummon_offset;
            YGO_SetResponsei(pduel_, (idx_ << 16) + 1);
          } else if (act == ActionAct::Repo) {
            uint32_t idx_ = idx - repos_offset;
            YGO_SetResponsei(pduel_, (idx_ << 16) + 2);
          } else if (act == ActionAct::MSet) {
            uint32_t idx_ = idx - mset_offset;
            YGO_SetResponsei(pduel_, (idx_ << 16) + 3);
          } else if (act == ActionAct::Set) {
            uint32_t idx_ = idx - set_offset;
            YGO_SetResponsei(pduel_, (idx_ << 16) + 4);
          } else if (act == ActionAct::Activate) {
            uint32_t idx_ = idx - activate_offset;
            YGO_SetResponsei(pduel_, (idx_ << 16) + 5);
          }
        }
      };
    } else if (msg_ == MSG_SELECT_PLACE || msg_ == MSG_SELECT_DISFIELD) {
      // TODO(1): add card informaton to select place
      auto player = read_u8();
      auto count = read_u8();
      if (count == 0) {
        count = 1;
      }
      if (count != 1) {
        auto s = fmt::format("Select place count {} not implemented for {}",
                              count, msg_ == MSG_SELECT_PLACE ? "place" : "disfield");
        throw std::runtime_error(s);
      }
      auto flag = read_u32();
      auto places = flag_to_usable_places(flag);
      if (verbose_) {
        auto place_s = msg_ == MSG_SELECT_PLACE ? "place" : "disfield";
        auto s = fmt::format("Select {} for card, one of:", place_s);
        players_[player]->notify(s);
      }
      for (int i = 0; i < places.size(); ++i) {
        legal_actions_.push_back(LegalAction::place(places[i]));
        if (verbose_) {
          auto s = fmt::format("{}: {}", i + 1, action_place_to_string(places[i]));
          players_[player]->notify(s);
        }
      }
      to_play_ = player;
      callback_ = [this, player](int idx) {
        auto place = legal_actions_[idx].place_;
        int i = static_cast<int>(place);
        uint8_t plr = player;
        uint8_t loc;
        uint8_t seq;
        if (
          i >= static_cast<int>(ActionPlace::MZone1) &&
          i <= static_cast<int>(ActionPlace::MZone7)) {
          loc = LOCATION_MZONE;
          seq = i - static_cast<int>(ActionPlace::MZone1);
        } else if (
          i >= static_cast<int>(ActionPlace::SZone1) &&
          i <= static_cast<int>(ActionPlace::SZone8)) {
          loc = LOCATION_SZONE;
          seq = i - static_cast<int>(ActionPlace::SZone1);
        } else if (
          i >= static_cast<int>(ActionPlace::OpMZone1) &&
          i <= static_cast<int>(ActionPlace::OpMZone7)) {
          plr = 1 - player;
          loc = LOCATION_MZONE;
          seq = i - static_cast<int>(ActionPlace::OpMZone1);
        } else if (
          i >= static_cast<int>(ActionPlace::OpSZone1) &&
          i <= static_cast<int>(ActionPlace::OpSZone8)) {
          plr = 1 - player;
          loc = LOCATION_SZONE;
          seq = i - static_cast<int>(ActionPlace::OpSZone1);
        }
        resp_buf_[0] = plr;
        resp_buf_[1] = loc;
        resp_buf_[2] = seq;
        YGO_SetResponseb(pduel_, resp_buf_);
      };
    } else if (msg_ == MSG_SELECT_COUNTER) {
      auto player = read_u8();
      auto counter_type = read_u16();
      int counter_count = read_u16();
      int count = read_u8();
      if (count > 2) {
        throw std::runtime_error("Select counter count " +
                                 std::to_string(count) + " not implemented");
      }
      auto& pl = players_[player];
      if (verbose_) {
        pl->notify(fmt::format("Type new {} for {} card(s), separated by spaces.", "UNKNOWN_COUNTER", count));
      }
      std::vector<int> counters;
      counters.reserve(count);
      for (int i = 0; i < count; ++i) {
        auto code = read_u32();
        auto controller = read_u8();
        auto loc = read_u8();
        auto seq = read_u8();
        auto counter = read_u16();
        counters.push_back(counter & 0xffff);

        if (verbose_) {
          pl->notify(c_get_card(code).name_ + ": " + std::to_string(counter));
        }
        // auto spec = ls_to_spec(loc, seq, 0, controller != player);
        // options_.push_back(spec);
      }
      // TODO(2): implement action
      n_counters_ = count;
      uint16_t resp1 = static_cast<uint16_t>(std::min(counter_count, counters[0]));
      memcpy(resp_buf_, &resp1, 2);
      counter_count -= counters[0];
      if (count == 2) {
        uint16_t resp2 = 0;
        if (counter_count > 0) {
          resp2 = static_cast<uint16_t>(counter_count);
        }
        memcpy(resp_buf_ + 2, &resp2, 2);
      }
      YGO_SetResponseb(pduel_, resp_buf_);
    } else if (msg_ == MSG_ANNOUNCE_RACE) {
      auto player = read_u8();
      const int count = read_u8();
      const uint32_t available = read_u32();
      if (count != 1) {
        throw std::runtime_error(
            "Announce race count " + std::to_string(count) +
            " not implemented");
      }

      for (const auto &[race, name] : race2str) {
        if (race == RACE_NONE || (available & race) == 0) {
          continue;
        }
        LegalAction action = LegalAction::number(race_to_id(race));
        action.response_ = race;
        legal_actions_.push_back(action);
      }
      if (legal_actions_.empty()) {
        throw std::runtime_error(fmt::format(
            "Announce race has no legal choices for mask 0x{:08x}",
            available));
      }

      if (verbose_) {
        auto& pl = players_[player];
        pl->notify("Select 1 race:");
        for (int i = 0; i < legal_actions_.size(); ++i) {
          const auto race = legal_actions_[i].response_;
          pl->notify(fmt::format(
              "{}: {}", i + 1, race2str.at(race)));
        }
      }

      to_play_ = player;
      callback_ = [this](int idx) {
        YGO_SetResponsei(pduel_, legal_actions_[idx].response_);
      };
    } else if (msg_ == MSG_ANNOUNCE_NUMBER) {
      auto player = read_u8();
      int count = read_u8();
      std::vector<uint32_t> numbers;
      for (int i = 0; i < count; ++i) {
        const uint32_t number = read_u32();
        numbers.push_back(number);
        LegalAction action = LegalAction::number(
            announce_number_to_id(number));
        action.response_ = number;
        legal_actions_.push_back(action);
      }
      if (verbose_) {
        auto& pl = players_[player];
        std::string str = "Select a number, one of:";
        pl->notify(str);
        for (int i = 0; i < count; ++i) {
          pl->notify(fmt::format("{}: {}", i + 1, numbers[i]));
        }
      }
      to_play_ = player;
      callback_ = [this](int idx) {
        YGO_SetResponsei(pduel_, idx);
      };
    } else if (msg_ == MSG_ANNOUNCE_ATTRIB) {
      auto player = read_u8();
      int count = read_u8();
      auto flag = read_u32();

      int n_attrs = 7;

      std::vector<uint8_t> attrs;
      for (int i = 0; i < n_attrs; i++) {
        if (flag & (1 << i)) {
          attrs.push_back(i + 1);
        }
      }
      // TODO(2): implement action
      if (count != 1) {
        throw std::runtime_error("Announce attrib count " +
                                 std::to_string(count) + " not implemented");
      }

      if (verbose_) {
        auto& pl = players_[player];
        pl->notify("Select " + std::to_string(count) +
                   " attributes separated by spaces:");
        for (int i = 0; i < attrs.size(); i++) {
          pl->notify(fmt::format("{}: {}", i + 1, attribute_to_string(1 << (attrs[i] - 1))));
        }
      }

      // auto combs = combinations(attrs.size(), count);
      for (int i = 0; i < attrs.size(); i++) {
        legal_actions_.push_back(LegalAction::attribute(1 << (attrs[i] - 1)));
      }

      to_play_ = player;
      callback_ = [this](int idx) {
        const auto &action = legal_actions_[idx];
        uint32_t resp = 0;
        resp |= action.attribute_;
        YGO_SetResponsei(pduel_, resp);
      };
    } else if (msg_ == MSG_ANNOUNCE_CARD) {
      auto player = read_u8();
      int count = read_u8();

      std::vector<uint32_t> opcodes;
      opcodes.reserve(count);
      for (int i = 0; i < count; i++) {
        opcodes.push_back(read_u32());
      }

      auto codes = announce_card_candidates(
          opcodes, static_cast<size_t>(spec_.config["max_options"_]));

      if (verbose_) {
        auto& pl = players_[player];
        pl->notify("Select 1 card from the following cards:");
        for (int i = 0; i < codes.size(); i++) {
          pl->notify(fmt::format("{}: {}", i + 1, c_get_card(codes[i]).name_));
        }
      }

      for (auto code : codes) {
        LegalAction la;
        la.cid_ = c_get_card_id(code);
        la.response_ = code;
        legal_actions_.push_back(la);
      }

      to_play_ = player;
      callback_ = [this](int idx) {
        const auto &action = legal_actions_[idx];
        uint32_t resp = action.response_;
        YGO_SetResponsei(pduel_, resp);
      };
    } else if (msg_ == MSG_SELECT_POSITION) {
      auto player = read_u8();
      auto code = read_u32();
      auto valid_pos = read_u8();
      CardId cid = c_get_card_id(code);

      if (verbose_) {
        auto& pl = players_[player];
        auto card = c_get_card(code);
        pl->notify("Select position for " + card.name_ + ":");
      }

      for (auto pos : {POS_FACEUP_ATTACK, POS_FACEDOWN_ATTACK,
                       POS_FACEUP_DEFENSE, POS_FACEDOWN_DEFENSE}) {
        if (valid_pos & pos) {
          LegalAction la;
          la.cid_ = cid;
          la.position_ = pos;
          legal_actions_.push_back(la);
          int cmd_idx = legal_actions_.size();
          if (verbose_) {
            auto& pl = players_[player];
            pl->notify(fmt::format("{}: {}", cmd_idx, position_to_string(pos)));
          }
        }
      }

      to_play_ = player;
      callback_ = [this](int idx) {
        uint8_t pos = legal_actions_[idx].position_;
        YGO_SetResponsei(pduel_, pos);
      };
    } else {
      show_deck(0);
      show_deck(1);
      show_buffer();
      throw std::runtime_error(
        fmt::format("Unknown message {}, length {}, dp {}",
        msg_to_string(msg_), dl_, dp_));
    }
  }

  void _damage(uint8_t player, uint32_t amount) {
    lp_[player] -= amount;
    if (verbose_) {
      auto& lp = players_[player];
      lp->notify(fmt::format("Your lp decreased by {}, now {}", amount, lp_[player]));
      players_[1 - player]->notify(fmt::format("{}'s lp decreased by {}, now {}",
                                   lp->nickname_, amount, lp_[player]));
    }
  }

  void _recover(uint8_t player, uint32_t amount) {
    lp_[player] += amount;
    if (verbose_) {
      auto& lp = players_[player];
      lp->notify(fmt::format("Your lp increased by {}, now {}", amount, lp_[player]));
      players_[1 - player]->notify(fmt::format("{}'s lp increased by {}, now {}",
                                   lp->nickname_, amount, lp_[player]));
    }
  }

  void _duel_end(uint8_t player, uint8_t reason) {
    winner_ = player;
    win_reason_ = reason;
    YGO_EndDuel(pduel_);

    duel_started_ = false;
  }
};

class YGOProEnv : public Env<YGOProEnvSpec> {
protected:
  const int max_episode_steps_;
  const int timeout_;

  int elapsed_step_;

  std::uniform_int_distribution<uint64_t> dist_int_;

  // The pool can't be in vector, so we create multiple pools manually
  BS::thread_pool pool0_;
  BS::thread_pool pool1_;
  BS::thread_pool pool2_;
  BS::thread_pool pool3_;
  BS::thread_pool pool4_;

  const int max_timeout_{5};
 
  // YGOProEnvImpl env_impl0_;
  // YGOProEnvImpl env_impl1_;
  // YGOProEnvImpl env_impl2_;
  // YGOProEnvImpl env_impl3_;
  // YGOProEnvImpl env_impl4_;
  std::vector<YGOProEnvImpl> env_impls_;

  bool done_{true};

public:
  YGOProEnv(const Spec &spec, int env_id)
      : Env<YGOProEnvSpec>(spec, env_id),
        max_episode_steps_(spec.config["max_episode_steps"_]),
        elapsed_step_(max_episode_steps_ + 1),
        timeout_(spec.config["timeout"_]),
        pool0_(1), pool1_(1), pool2_(1), pool3_(1), pool4_(1),
        dist_int_(0, 0xffffffff) {
    env_impls_.reserve(max_timeout_);
    env_impls_.emplace_back(spec, dist_int_(gen_));
  }

  bool IsDone() override { return done_; }

  BS::thread_pool& get_pool(int idx) {
    switch (idx) {
      case 0: return pool0_;
      case 1: return pool1_;
      case 2: return pool2_;
      case 3: return pool3_;
      case 4: return pool4_;
      default: throw std::runtime_error("Invalid pool index");
    }
  }

  void handle_timeout() {
    if (env_impls_.size() >= static_cast<size_t>(max_timeout_)) {
      throw std::runtime_error("Too many timeouts");
    }
    env_impls_.emplace_back(spec_, dist_int_(gen_));
    done_ = true;
    State state = Allocate();
    state["reward"_] = 0.0;
    state["info:to_play"_] = 0;
    state["info:is_selfplay"_] = 1;
    state["info:win_reason"_] = 0;
    state["info:num_options"_] = 1;
    state["obs:global_"_][22] = uint8_t(1);
  }

  void Reset() override {
    int idx = env_impls_.size() - 1;
    auto& pool = get_pool(idx);
    auto fut = pool.submit_task([this, idx]() {
      env_impls_[idx].reset();
    });
    if (fut.wait_for(std::chrono::seconds(timeout_)) != std::future_status::ready) {
      throw std::runtime_error("Reset timeout");
    }

    auto &env_impl = env_impls_[idx];
    elapsed_step_ = 0;
    done_ = false;
    State state = Allocate();
    env_impl.WriteState(state);
  }

  void Step(const Action &action) override {
    int idx = env_impls_.size() - 1;
    auto& pool = get_pool(idx);
    int action_idx = action["action"_];
    pool.detach_task([this, action_idx, idx]() {
      // Test timeout: random sleep with probability 0.01
      // if (dist_int_(gen_) % 10000 == 0) {
      //   fmt::println("Env {} sleep {}", env_id_, env_impls_.capacity());
      //   std::this_thread::sleep_for(std::chrono::seconds(5));
      //   fmt::println("Env {} after {}", env_id_, env_impls_.capacity());
      //   auto& env_impl = env_impls_[idx];
      //   env_impl.step(action_idx);
      //   std::this_thread::sleep_for(std::chrono::seconds(1));
      //   return;
      // }
      env_impls_[idx].step(action_idx);
    });
    if (!pool.wait_for(std::chrono::seconds(timeout_))) {
      handle_timeout();
      fmt::println("Env {} timeout, new env created", env_id_);
    } else {
      auto& env_impl = env_impls_[idx];
      done_ = env_impl.done();
      State state = Allocate();
      env_impl.WriteState(state);
    }
  }

};

using YGOProEnvPool = AsyncEnvPool<YGOProEnv>;

} // namespace ygopro

template <>
struct fmt::formatter<ygopro::LegalAction>: formatter<string_view> {

    // Format the LegalAction object
    template <typename FormatContext>
    auto format(const ygopro::LegalAction& action, FormatContext& ctx) const {
        std::stringstream ss;
        ss << "{";
        if (!action.spec_.empty()) {
          ss << "spec='" << action.spec_ << "', ";
        }
        if (action.cid_ != 0) {
          ss << "cid=" << action.cid_ << ", ";
        }
        if (action.act_ != ygopro::ActionAct::None) {
          ss << "act=" << ygopro::action_act_to_string(action.act_) << ", ";
        }
        if (action.phase_ != ygopro::ActionPhase::None) {
          ss << "phase=" << ygopro::action_phase_to_string(action.phase_) << ", ";
        }
        if (action.finish_) {
          ss << "finish=true, ";
        }
        if (action.position_ != 0) {
          ss << "position=" << ygopro::position_to_string(action.position_) << ", ";
        }
        if (action.effect_ != -1) {
          ss << "effect=" << action.effect_ << ", ";
        }
        if (action.number_ != 0) {
          ss << "number=" << int(action.number_) << ", ";
        }
        if (action.place_ != ygopro::ActionPlace::None) {
          ss << "place=" << ygopro::action_place_to_string(action.place_) << ", ";
        }
        if (action.attribute_ != 0) {
          ss << "attribute=" << ygopro::attribute_to_string(action.attribute_) << ", ";
        }
        std::string s = ss.str();
        if (s.back() == ' ') {
          s.pop_back();
          s.pop_back();
        }
        s.push_back('}');
        return format_to(ctx.out(), "{}", s);
    }
};

#endif // YGOENV_YGOPRO_YGOPRO_H_
