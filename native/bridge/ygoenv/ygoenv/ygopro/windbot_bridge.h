#ifndef YGOENV_YGOPRO_WINDBOT_BRIDGE_H_
#define YGOENV_YGOPRO_WINDBOT_BRIDGE_H_

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

extern char **environ;

namespace ygopro {

struct WindBotBridgeConfig {
  std::string root;
  std::string executable;
  std::string executor;
  std::string deck_file;
  std::string cards_db;
  std::string log_path;
  int seed = 1;
  int timeout_ms = 30000;
  bool verbose = false;
};

class WindBotBridge {
 public:
  WindBotBridge(
      const WindBotBridgeConfig &config, uint8_t player,
      intptr_t pduel, const std::array<int, 2> &main_counts,
      const std::array<int, 2> &extra_counts)
      : config_(config), player_(player), pduel_(pduel) {
    if (player_ > 1) {
      throw std::runtime_error("WindBot player must be 0 or 1");
    }
    try {
      start(main_counts, extra_counts);
    } catch (...) {
      stop();
      throw;
    }
  }

  WindBotBridge(const WindBotBridge &) = delete;
  WindBotBridge &operator=(const WindBotBridge &) = delete;

  ~WindBotBridge() { stop(); }

  std::optional<std::vector<uint8_t>> Forward(
      const uint8_t *message, size_t length) {
    if (length == 0 || finished_) {
      return std::nullopt;
    }
    const int msg = message[0];
    const auto decision_player = DecisionPlayer(message, length);
    if (decision_player.has_value()) {
      last_decision_for_bot_ = *decision_player == player_;
      if (!last_decision_for_bot_) {
        const uint8_t waiting = 3;
        SendPacket(kStocGameMsg, &waiting, 1);
        return std::nullopt;
      }
      RefreshAll();
      auto filtered = FilterMessage(message, length);
      if (!filtered.has_value()) {
        throw std::runtime_error(
            "WindBot decision message was filtered unexpectedly");
      }
      SendPacket(kStocGameMsg, filtered->data(), filtered->size());
      return ReceiveResponse();
    }

    if (msg == MSG_RETRY) {
      if (!last_decision_for_bot_) {
        return std::nullopt;
      }
      SendPacket(kStocGameMsg, message, length);
      return ReceiveResponse();
    }

    auto filtered = FilterMessage(message, length);
    if (filtered.has_value()) {
      SendPacket(kStocGameMsg, filtered->data(), filtered->size());
    }
    if (msg == MSG_WIN) {
      Finish();
    }
    return std::nullopt;
  }

  void Finish() {
    if (finished_) {
      return;
    }
    finished_ = true;
    if (client_fd_ >= 0) {
      try {
        SendPacket(kStocDuelEnd, nullptr, 0);
      } catch (...) {
      }
    }
  }

 private:
  static constexpr uint8_t kCtosResponse = 0x01;
  static constexpr uint8_t kCtosUpdateDeck = 0x02;
  static constexpr uint8_t kCtosPlayerInfo = 0x10;
  static constexpr uint8_t kCtosJoinGame = 0x12;
  static constexpr uint8_t kCtosExternalAddress = 0x17;
  static constexpr uint8_t kCtosHsReady = 0x22;
  static constexpr uint8_t kCtosHsStart = 0x25;

  static constexpr uint8_t kStocGameMsg = 0x01;
  static constexpr uint8_t kStocDeckCount = 0x09;
  static constexpr uint8_t kStocJoinGame = 0x12;
  static constexpr uint8_t kStocTypeChange = 0x13;
  static constexpr uint8_t kStocDuelStart = 0x15;
  static constexpr uint8_t kStocDuelEnd = 0x16;
  static constexpr uint8_t kStocHsPlayerEnter = 0x20;
  static constexpr uint8_t kStocHsPlayerChange = 0x21;

  static constexpr uint8_t kMsgStart = 4;
  static constexpr uint8_t kMsgUpdateData = 6;
  static constexpr uint8_t kMsgCardSelected = 80;

  struct Packet {
    uint8_t protocol = 0;
    std::vector<uint8_t> payload;
  };

  WindBotBridgeConfig config_;
  uint8_t player_ = 0;
  intptr_t pduel_ = 0;
  int listener_fd_ = -1;
  int client_fd_ = -1;
  pid_t child_pid_ = -1;
  bool finished_ = false;
  bool last_decision_for_bot_ = false;

  static uint16_t ReadU16(const uint8_t *data) {
    return static_cast<uint16_t>(data[0])
        | (static_cast<uint16_t>(data[1]) << 8);
  }

  static uint32_t ReadU32(const uint8_t *data) {
    return static_cast<uint32_t>(data[0])
        | (static_cast<uint32_t>(data[1]) << 8)
        | (static_cast<uint32_t>(data[2]) << 16)
        | (static_cast<uint32_t>(data[3]) << 24);
  }

  static void WriteU16(uint8_t *data, uint16_t value) {
    data[0] = static_cast<uint8_t>(value);
    data[1] = static_cast<uint8_t>(value >> 8);
  }

  static void WriteU32(uint8_t *data, uint32_t value) {
    data[0] = static_cast<uint8_t>(value);
    data[1] = static_cast<uint8_t>(value >> 8);
    data[2] = static_cast<uint8_t>(value >> 16);
    data[3] = static_cast<uint8_t>(value >> 24);
  }

  static void WriteUtf16(
      uint8_t *destination, size_t code_units, const std::string &text) {
    std::memset(destination, 0, code_units * 2);
    const size_t count = std::min(code_units - 1, text.size());
    for (size_t i = 0; i < count; ++i) {
      destination[i * 2] = static_cast<uint8_t>(text[i]);
    }
  }

  static std::string ResolveExecutable(const WindBotBridgeConfig &config) {
    if (!config.executable.empty()) {
      return config.executable;
    }
    return (std::filesystem::path(config.root) / "WindBot.exe").string();
  }

  void start(
      const std::array<int, 2> &main_counts,
      const std::array<int, 2> &extra_counts) {
    ValidateConfig();
    CreateListener();
    SpawnWindBot();
    AcceptWindBot();
    Handshake();
    SendDuelStart(main_counts, extra_counts);
  }

  void ValidateConfig() const {
    for (const auto &[name, value] : std::array{
             std::pair{"windbot_root", config_.root},
             std::pair{"windbot_executor", config_.executor},
             std::pair{"windbot_deck_file", config_.deck_file},
             std::pair{"windbot_cards_db", config_.cards_db}}) {
      if (value.empty()) {
        throw std::runtime_error(std::string(name) + " is required");
      }
    }
    if (config_.timeout_ms <= 0) {
      throw std::runtime_error("windbot_timeout_ms must be positive");
    }
    const auto executable = ResolveExecutable(config_);
    for (const auto &[name, value] : std::array{
             std::pair{"WindBot executable", executable},
             std::pair{"WindBot deck file", config_.deck_file},
             std::pair{"WindBot cards database", config_.cards_db}}) {
      if (!std::filesystem::is_regular_file(value)) {
        throw std::runtime_error(
            std::string(name) + " does not exist: " + value);
      }
    }
  }

  void CreateListener() {
    listener_fd_ = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (listener_fd_ < 0) {
      ThrowSystemError("socket");
    }
    int reuse = 1;
    setsockopt(
        listener_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    if (bind(
            listener_fd_, reinterpret_cast<sockaddr *>(&address),
            sizeof(address)) != 0) {
      ThrowSystemError("bind");
    }
    if (listen(listener_fd_, 1) != 0) {
      ThrowSystemError("listen");
    }
  }

  uint16_t ListenerPort() const {
    sockaddr_in address{};
    socklen_t length = sizeof(address);
    if (getsockname(
            listener_fd_, reinterpret_cast<sockaddr *>(&address),
            &length) != 0) {
      ThrowSystemError("getsockname");
    }
    return ntohs(address.sin_port);
  }

  void SpawnWindBot() {
    const auto port = ListenerPort();
    const auto executable = ResolveExecutable(config_);
    const auto log_path = config_.log_path.empty()
        ? fmt::format("/tmp/ygoenv-windbot-{}-{}.log", getpid(), port)
        : fmt::format("{}.{}.log", config_.log_path, port);
    std::vector<std::string> arguments{
        "mono",
        executable,
        fmt::format("Name=YGOEnv-WindBot-{}", port),
        "Host=127.0.0.1",
        fmt::format("Port={}", port),
        "HostInfo=",
        "Version=4961",
        fmt::format("Deck={}", config_.executor),
        fmt::format("DeckFile={}", config_.deck_file),
        "Hand=1",
        "SelectFirst=true",
        "Chat=false",
        fmt::format("Debug={}", config_.verbose ? "true" : "false"),
        fmt::format("DbPath={}", config_.cards_db),
        "A35=false",
        fmt::format("Seed={}", config_.seed),
    };
    std::vector<char *> argv;
    argv.reserve(arguments.size() + 1);
    for (auto &argument : arguments) {
      argv.push_back(argument.data());
    }
    argv.push_back(nullptr);

    posix_spawn_file_actions_t actions;
    if (posix_spawn_file_actions_init(&actions) != 0) {
      throw std::runtime_error("posix_spawn_file_actions_init failed");
    }
    int error = posix_spawn_file_actions_addopen(
        &actions, STDOUT_FILENO, log_path.c_str(),
        O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (error == 0) {
      error = posix_spawn_file_actions_adddup2(
          &actions, STDOUT_FILENO, STDERR_FILENO);
    }
    if (error == 0) {
      error = posix_spawn_file_actions_addchdir_np(
          &actions, config_.root.c_str());
    }
    if (error == 0) {
      error = posix_spawnp(
          &child_pid_, "mono", &actions, nullptr, argv.data(), environ);
    }
    posix_spawn_file_actions_destroy(&actions);
    if (error != 0) {
      throw std::runtime_error(
          "failed to spawn WindBot: " + std::string(std::strerror(error)));
    }
  }

  void AcceptWindBot() {
    pollfd descriptor{listener_fd_, POLLIN, 0};
    const int ready = poll(&descriptor, 1, config_.timeout_ms);
    if (ready <= 0) {
      if (ready == 0) {
        throw std::runtime_error("timed out waiting for WindBot connection");
      }
      ThrowSystemError("poll(listener)");
    }
    client_fd_ = accept4(listener_fd_, nullptr, nullptr, SOCK_CLOEXEC);
    if (client_fd_ < 0) {
      ThrowSystemError("accept4");
    }
    close(listener_fd_);
    listener_fd_ = -1;
  }

  void Handshake() {
    bool joined = false;
    while (!joined) {
      const auto packet = ReceivePacket();
      if (packet.protocol == kCtosJoinGame) {
        joined = true;
      } else if (
          packet.protocol != kCtosPlayerInfo
          && packet.protocol != kCtosExternalAddress) {
        throw std::runtime_error(fmt::format(
            "unexpected WindBot handshake packet: 0x{:02x}",
            packet.protocol));
      }
    }

    std::array<uint8_t, 20> host_info{};
    host_info[6] = 5;
    host_info[7] = 1;
    host_info[8] = 1;
    WriteU32(host_info.data() + 12, 8000);
    host_info[16] = 5;
    host_info[17] = 1;
    SendPacket(kStocJoinGame, host_info.data(), host_info.size());

    const uint8_t type = static_cast<uint8_t>(0x10 | player_);
    SendPacket(kStocTypeChange, &type, 1);
    SendPlayerEnter("JAX-Agent", 1 - player_);
    SendPlayerEnter("YGOEnv-WindBot", player_);

    bool deck_received = false;
    bool ready_received = false;
    while (!deck_received || !ready_received) {
      const auto packet = ReceivePacket();
      if (packet.protocol == kCtosUpdateDeck) {
        deck_received = true;
      } else if (packet.protocol == kCtosHsReady) {
        ready_received = true;
      } else {
        throw std::runtime_error(fmt::format(
            "unexpected WindBot ready packet: 0x{:02x}",
            packet.protocol));
      }
    }

    const uint8_t player0_ready = 0x09;
    const uint8_t player1_ready = 0x19;
    SendPacket(kStocHsPlayerChange, &player0_ready, 1);
    SendPacket(kStocHsPlayerChange, &player1_ready, 1);
  }

  void SendPlayerEnter(const std::string &name, uint8_t position) {
    std::array<uint8_t, 41> payload{};
    WriteUtf16(payload.data(), 20, name);
    payload[40] = position;
    SendPacket(kStocHsPlayerEnter, payload.data(), payload.size());
  }

  void SendDuelStart(
      const std::array<int, 2> &main_counts,
      const std::array<int, 2> &extra_counts) {
    SendPacket(kStocDuelStart, nullptr, 0);

    std::array<uint8_t, 12> deck_counts{};
    const uint8_t local = player_;
    const uint8_t enemy = 1 - player_;
    WriteU16(deck_counts.data(), main_counts[local]);
    WriteU16(deck_counts.data() + 2, extra_counts[local]);
    WriteU16(deck_counts.data() + 4, 0);
    WriteU16(deck_counts.data() + 6, main_counts[enemy]);
    WriteU16(deck_counts.data() + 8, extra_counts[enemy]);
    WriteU16(deck_counts.data() + 10, 0);
    SendPacket(
        kStocDeckCount, deck_counts.data(), deck_counts.size());

    std::array<uint8_t, 19> start{};
    start[0] = kMsgStart;
    start[1] = player_;
    start[2] = 5;
    WriteU32(start.data() + 3, 8000);
    WriteU32(start.data() + 7, 8000);
    WriteU16(start.data() + 11, main_counts[0]);
    WriteU16(start.data() + 13, extra_counts[0]);
    WriteU16(start.data() + 15, main_counts[1]);
    WriteU16(start.data() + 17, extra_counts[1]);
    SendPacket(kStocGameMsg, start.data(), start.size());
  }

  static bool IsDecisionMessage(int msg) {
    switch (msg) {
      case MSG_SELECT_BATTLECMD:
      case MSG_SELECT_IDLECMD:
      case MSG_SELECT_EFFECTYN:
      case MSG_SELECT_YESNO:
      case MSG_SELECT_OPTION:
      case MSG_SELECT_CARD:
      case MSG_SELECT_UNSELECT_CARD:
      case MSG_SELECT_CHAIN:
      case MSG_SELECT_PLACE:
      case MSG_SELECT_POSITION:
      case MSG_SELECT_TRIBUTE:
      case MSG_SELECT_COUNTER:
      case MSG_SELECT_SUM:
      case MSG_SELECT_DISFIELD:
      case MSG_SORT_CARD:
      case MSG_ROCK_PAPER_SCISSORS:
      case MSG_ANNOUNCE_RACE:
      case MSG_ANNOUNCE_ATTRIB:
      case MSG_ANNOUNCE_CARD:
      case MSG_ANNOUNCE_NUMBER:
        return true;
      default:
        return false;
    }
  }

  static std::optional<uint8_t> DecisionPlayer(
      const uint8_t *message, size_t length) {
    if (length < 2 || !IsDecisionMessage(message[0])) {
      return std::nullopt;
    }
    if (message[0] == MSG_SELECT_SUM) {
      if (length < 3) {
        throw std::runtime_error("truncated MSG_SELECT_SUM");
      }
      return message[2];
    }
    return message[1];
  }

  std::optional<std::vector<uint8_t>> FilterMessage(
      const uint8_t *message, size_t length) const {
    std::vector<uint8_t> result(message, message + length);
    const int msg = result[0];
    if (msg == MSG_HINT) {
      if (length < 7) {
        throw std::runtime_error("truncated MSG_HINT");
      }
      const uint8_t type = result[1];
      const uint8_t player = result[2];
      const bool visible =
          ((type == 1 || type == 2 || type == 3 || type == 5)
           && player == player_)
          || ((type == 4 || type == 6 || type == 7 || type == 8
               || type == 9 || type == 11)
              && player != player_)
          || type == 10;
      if (!visible) {
        return std::nullopt;
      }
    } else if (
        msg == MSG_SELECT_CARD || msg == MSG_SELECT_TRIBUTE) {
      FilterSelectCards(result, 6);
    } else if (msg == MSG_SELECT_UNSELECT_CARD) {
      if (length < 7) {
        throw std::runtime_error("truncated MSG_SELECT_UNSELECT_CARD");
      }
      size_t cursor = 7;
      const uint8_t select_count = result[6];
      FilterCardEntries(result, cursor, select_count, result[1]);
      cursor += static_cast<size_t>(select_count) * 8;
      if (cursor >= result.size()) {
        throw std::runtime_error("truncated MSG_SELECT_UNSELECT_CARD");
      }
      const uint8_t unselect_count = result[cursor++];
      FilterCardEntries(result, cursor, unselect_count, result[1]);
    } else if (msg == MSG_CONFIRM_CARDS) {
      if (length < 9) {
        throw std::runtime_error("truncated MSG_CONFIRM_CARDS");
      }
      const uint8_t target = result[1];
      const uint8_t location = result[8];
      if (location == LOCATION_DECK && target != player_) {
        return std::nullopt;
      }
    } else if (msg == MSG_SHUFFLE_HAND || msg == MSG_SHUFFLE_EXTRA) {
      if (length < 3) {
        throw std::runtime_error("truncated shuffle message");
      }
      const uint8_t owner = result[1];
      const size_t count = result[2];
      if (length < 3 + count * 4) {
        throw std::runtime_error("truncated shuffle card list");
      }
      if (owner != player_) {
        std::memset(result.data() + 3, 0, count * 4);
      }
    } else if (msg == MSG_MOVE) {
      if (length < 17) {
        throw std::runtime_error("truncated MSG_MOVE");
      }
      const uint8_t current_controller = result[9];
      const uint8_t current_location = result[10];
      const uint8_t current_position = result[12];
      const bool hidden =
          !(current_location & (LOCATION_GRAVE | LOCATION_OVERLAY))
          && ((current_location & (LOCATION_DECK | LOCATION_HAND))
              || (current_position & POS_FACEDOWN));
      if (current_controller != player_ && hidden) {
        WriteU32(result.data() + 1, 0);
      }
    } else if (msg == MSG_SET) {
      if (length < 9) {
        throw std::runtime_error("truncated MSG_SET");
      }
      WriteU32(result.data() + 1, 0);
    } else if (msg == MSG_DRAW) {
      if (length < 3) {
        throw std::runtime_error("truncated MSG_DRAW");
      }
      const uint8_t owner = result[1];
      const size_t count = result[2];
      if (length < 3 + count * 4) {
        throw std::runtime_error("truncated MSG_DRAW card list");
      }
      if (owner != player_) {
        for (size_t i = 0; i < count; ++i) {
          auto *code = result.data() + 3 + i * 4;
          if ((code[3] & 0x80) == 0) {
            WriteU32(code, 0);
          }
        }
      }
    } else if (msg == kMsgCardSelected) {
      return std::nullopt;
    } else if (msg == MSG_MISSED_EFFECT) {
      if (length < 2 || result[1] != player_) {
        return std::nullopt;
      }
    }
    return result;
  }

  static void FilterSelectCards(
      std::vector<uint8_t> &message, size_t entries_offset) {
    if (message.size() < entries_offset) {
      throw std::runtime_error("truncated select-card message");
    }
    const uint8_t player = message[1];
    const uint8_t count = message[entries_offset - 1];
    FilterCardEntries(message, entries_offset, count, player);
  }

  static void FilterCardEntries(
      std::vector<uint8_t> &message, size_t offset,
      size_t count, uint8_t player) {
    if (message.size() < offset + count * 8) {
      throw std::runtime_error("truncated card entries");
    }
    for (size_t i = 0; i < count; ++i) {
      auto *entry = message.data() + offset + i * 8;
      if (entry[4] != player) {
        WriteU32(entry, 0);
      }
    }
  }

  void RefreshAll() {
    for (uint8_t owner = 0; owner < 2; ++owner) {
      RefreshZone(owner, LOCATION_MZONE);
      RefreshZone(owner, LOCATION_SZONE);
      RefreshZone(owner, LOCATION_HAND);
      RefreshZone(owner, LOCATION_GRAVE);
      RefreshZone(owner, LOCATION_EXTRA);
      RefreshZone(owner, LOCATION_REMOVED);
    }
  }

  void RefreshZone(uint8_t owner, uint8_t location) {
    constexpr int kFullQuery = 0xefffff;
    constexpr size_t kQueryCapacity = 0x20000;
    std::vector<uint8_t> payload(3 + kQueryCapacity);
    payload[0] = kMsgUpdateData;
    payload[1] = owner;
    payload[2] = location;
    const int length = query_field_card(
        pduel_, owner, location, kFullQuery, payload.data() + 3, 0);
    if (length < 0 || static_cast<size_t>(length) > kQueryCapacity) {
      throw std::runtime_error("invalid query_field_card response length");
    }
    payload.resize(static_cast<size_t>(length) + 3);
    if (owner != player_ && location != LOCATION_GRAVE) {
      HideOpponentQuery(payload.data() + 3, length, location);
    }
    SendPacket(kStocGameMsg, payload.data(), payload.size());
  }

  static void HideOpponentQuery(
      uint8_t *query, int length, uint8_t location) {
    int consumed = 0;
    auto *cursor = query;
    while (consumed < length) {
      if (length - consumed < 4) {
        throw std::runtime_error("truncated query chunk length");
      }
      const int chunk_length = static_cast<int>(ReadU32(cursor));
      if (chunk_length < 4 || chunk_length > length - consumed) {
        throw std::runtime_error("invalid query chunk length");
      }
      auto *fields = cursor + 4;
      if (chunk_length > 12) {
        const uint32_t position_word = ReadU32(fields + 8);
        const uint8_t position =
            static_cast<uint8_t>(position_word >> 24);
        const bool hidden =
            location == LOCATION_HAND
            ? !(position & POS_FACEUP)
            : (position & POS_FACEDOWN);
        if (hidden) {
          HideQueryIdentity(fields, chunk_length - 4);
        }
      }
      cursor += chunk_length;
      consumed += chunk_length;
    }
  }

  static void HideQueryIdentity(uint8_t *fields, int length) {
    if (length < 4) {
      return;
    }
    const uint32_t flags = ReadU32(fields);
    auto *cursor = fields + 4;
    auto clear_u32 = [&]() {
      if (cursor + 4 > fields + length) {
        throw std::runtime_error("truncated query field");
      }
      WriteU32(cursor, 0);
      cursor += 4;
    };
    auto skip_u32 = [&]() {
      if (cursor + 4 > fields + length) {
        throw std::runtime_error("truncated query field");
      }
      cursor += 4;
    };
    if (flags & QUERY_CODE) clear_u32();
    if (flags & QUERY_POSITION) skip_u32();
    for (const auto flag : {
             QUERY_ALIAS, QUERY_TYPE, QUERY_LEVEL, QUERY_RANK,
             QUERY_ATTRIBUTE, QUERY_RACE, QUERY_ATTACK, QUERY_DEFENSE,
             QUERY_BASE_ATTACK, QUERY_BASE_DEFENSE, QUERY_REASON,
             QUERY_REASON_CARD, QUERY_EQUIP_CARD}) {
      if (flags & flag) clear_u32();
    }
    for (const auto flag : {
             QUERY_TARGET_CARD, QUERY_OVERLAY_CARD, QUERY_COUNTERS}) {
      if (!(flags & flag)) {
        continue;
      }
      if (cursor + 4 > fields + length) {
        throw std::runtime_error("truncated query list count");
      }
      const uint32_t count = ReadU32(cursor);
      cursor += 4;
      if (cursor + count * 4 > fields + length) {
        throw std::runtime_error("truncated query list");
      }
      std::memset(cursor, 0, count * 4);
      cursor += count * 4;
    }
    for (const auto flag : {
             QUERY_OWNER, QUERY_STATUS, QUERY_LSCALE, QUERY_RSCALE}) {
      if (flags & flag) clear_u32();
    }
    if (flags & QUERY_LINK) {
      clear_u32();
      clear_u32();
    }
  }

  std::vector<uint8_t> ReceiveResponse() {
    while (true) {
      auto packet = ReceivePacket();
      if (packet.protocol == kCtosResponse) {
        if (packet.payload.empty()
            || packet.payload.size() > SIZE_RETURN_VALUE) {
          throw std::runtime_error(fmt::format(
              "invalid WindBot response length: {}",
              packet.payload.size()));
        }
        return packet.payload;
      }
      if (packet.protocol == kCtosHsStart) {
        continue;
      }
      throw std::runtime_error(fmt::format(
          "unexpected WindBot packet while awaiting response: 0x{:02x}",
          packet.protocol));
    }
  }

  void SendPacket(
      uint8_t protocol, const uint8_t *payload, size_t payload_size) {
    if (client_fd_ < 0) {
      throw std::runtime_error("WindBot socket is closed");
    }
    if (payload_size > 0xfffe) {
      throw std::runtime_error("WindBot packet payload is too large");
    }
    std::vector<uint8_t> frame(payload_size + 3);
    WriteU16(frame.data(), static_cast<uint16_t>(payload_size + 1));
    frame[2] = protocol;
    if (payload_size > 0) {
      std::memcpy(frame.data() + 3, payload, payload_size);
    }
    SendAll(frame.data(), frame.size());
  }

  Packet ReceivePacket() {
    std::array<uint8_t, 2> header{};
    ReceiveAll(header.data(), header.size());
    const uint16_t length = ReadU16(header.data());
    if (length < 1) {
      throw std::runtime_error("invalid WindBot packet length");
    }
    std::vector<uint8_t> body(length);
    ReceiveAll(body.data(), body.size());
    Packet packet;
    packet.protocol = body[0];
    packet.payload.assign(body.begin() + 1, body.end());
    return packet;
  }

  void SendAll(const uint8_t *data, size_t length) {
    size_t offset = 0;
    while (offset < length) {
      const ssize_t sent = send(
          client_fd_, data + offset, length - offset, MSG_NOSIGNAL);
      if (sent < 0) {
        if (errno == EINTR) {
          continue;
        }
        ThrowSystemError("send");
      }
      if (sent == 0) {
        throw std::runtime_error("WindBot socket closed while sending");
      }
      offset += static_cast<size_t>(sent);
    }
  }

  void ReceiveAll(uint8_t *data, size_t length) {
    size_t offset = 0;
    while (offset < length) {
      pollfd descriptor{client_fd_, POLLIN, 0};
      const int ready = poll(&descriptor, 1, config_.timeout_ms);
      if (ready <= 0) {
        if (ready == 0) {
          throw std::runtime_error("timed out waiting for WindBot packet");
        }
        if (errno == EINTR) {
          continue;
        }
        ThrowSystemError("poll(client)");
      }
      const ssize_t received =
          recv(client_fd_, data + offset, length - offset, 0);
      if (received < 0) {
        if (errno == EINTR) {
          continue;
        }
        ThrowSystemError("recv");
      }
      if (received == 0) {
        throw std::runtime_error("WindBot socket closed");
      }
      offset += static_cast<size_t>(received);
    }
  }

  [[noreturn]] static void ThrowSystemError(const std::string &operation) {
    throw std::runtime_error(
        operation + " failed: " + std::string(std::strerror(errno)));
  }

  void stop() {
    Finish();
    if (client_fd_ >= 0) {
      shutdown(client_fd_, SHUT_RDWR);
      close(client_fd_);
      client_fd_ = -1;
    }
    if (listener_fd_ >= 0) {
      close(listener_fd_);
      listener_fd_ = -1;
    }
    if (child_pid_ > 0) {
      int status = 0;
      for (int i = 0; i < 20; ++i) {
        const pid_t result = waitpid(child_pid_, &status, WNOHANG);
        if (result == child_pid_ || result < 0) {
          child_pid_ = -1;
          return;
        }
        usleep(10000);
      }
      kill(child_pid_, SIGTERM);
      waitpid(child_pid_, &status, 0);
      child_pid_ = -1;
    }
  }
};

}  // namespace ygopro

#endif  // YGOENV_YGOPRO_WINDBOT_BRIDGE_H_
