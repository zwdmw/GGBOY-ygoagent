// Members of YGOProEnvImpl: project accepted external replies through the
// existing native callbacks without submitting a second reply to the engine.
struct BridgeSelection {
  int msg;
  PlayerId player;
  std::vector<LegalAction> actions;
  std::function<void(int)> callback;
  int index, mode, minimum, maximum, must;
  std::vector<std::string> specs;
  std::vector<std::vector<int>> combinations;
  decltype(ms_spec2idx_) spec_indices;
  std::vector<int> selected;
};

struct BridgeEvents {
  std::array<std::vector<PublicEvent>, 2> events;
  std::string source;
  PlayerId source_player;
  CardId source_cid;
  int source_effect;
};

std::optional<BridgeSelection> bridge_prompt_;
std::optional<BridgeEvents> bridge_pending_events_;
std::optional<std::vector<int>> bridge_pending_selected_;
std::vector<uint8_t> *bridge_capture_ = nullptr;

BridgeSelection bridge_selection() const {
  return {msg_, to_play_, legal_actions_, callback_,
          ms_idx_, ms_mode_, ms_min_, ms_max_, ms_must_,
          ms_specs_, ms_combs_, ms_spec2idx_, ms_r_idxs_};
}

void bridge_restore(const BridgeSelection &s) {
  msg_ = s.msg;
  to_play_ = s.player;
  legal_actions_ = s.actions;
  callback_ = s.callback;
  ms_idx_ = s.index;
  ms_mode_ = s.mode;
  ms_min_ = s.minimum;
  ms_max_ = s.maximum;
  ms_must_ = s.must;
  ms_specs_ = s.specs;
  ms_combs_ = s.combinations;
  ms_spec2idx_ = s.spec_indices;
  ms_r_idxs_ = s.selected;
}

BridgeEvents bridge_events() const {
  return {public_events_, pending_source_spec_, pending_source_spec_player_,
          pending_source_cid_, pending_source_effect_};
}

void bridge_restore(const BridgeEvents &s) {
  public_events_ = s.events;
  pending_source_spec_ = s.source;
  pending_source_spec_player_ = s.source_player;
  pending_source_cid_ = s.source_cid;
  pending_source_effect_ = s.source_effect;
}

size_t bridge_response_size(const byte *buf) const {
  switch (msg_) {
    case MSG_SORT_CARD: return 1;
    case MSG_SELECT_COUNTER: return 2 * n_counters_;
    case MSG_SELECT_PLACE:
    case MSG_SELECT_DISFIELD: return 3;
    default: return static_cast<size_t>(buf[0]) + 1;
  }
}

LegalAction bridge_action(int index) {
  auto action = legal_actions_.at(index);
  action.msg_ = msg_;
  if (action.cid_ == 0 && !action.spec_.empty()) {
    action.cid_ = spec_to_card_id(action.spec_, to_play_);
  }
  return action;
}

bool bridge_project_multi(const std::vector<uint8_t> &response,
                          std::vector<uint8_t> &captured) {
  if (response.empty() || response.size() != size_t(response[0]) + 1 ||
      int(response[0]) < ms_must_) {
    return false;
  }
  std::vector<int> remaining;
  for (size_t i = 1 + ms_must_; i < response.size(); ++i) {
    int index = response[i];
    if (index >= int(ms_specs_.size()) ||
        std::find(remaining.begin(), remaining.end(), index) != remaining.end()) {
      return false;
    }
    remaining.push_back(index);
  }
  for (int step = 0; step < 128; ++step) {
    int index = -1, selected = -1;
    for (int wanted : remaining) {
      for (size_t i = 0; i < legal_actions_.size(); ++i) {
        if (!legal_actions_[i].finish_ &&
            legal_actions_[i].spec_ == ms_specs_[wanted]) {
          index = static_cast<int>(i);
          selected = wanted;
          break;
        }
      }
      if (index != -1) break;
    }
    if (remaining.empty()) {
      for (size_t i = 0; i < legal_actions_.size(); ++i) {
        if (legal_actions_[i].finish_) {
          index = static_cast<int>(i);
          break;
        }
      }
    }
    if (index == -1) return false;
    const auto action = bridge_action(index);
    callback_(index);
    record_public_event(to_play_, action);
    if (selected != -1) {
      remaining.erase(std::find(remaining.begin(), remaining.end(), selected));
    }
    if (!captured.empty()) {
      if (!remaining.empty() || captured.size() != response.size() ||
          captured[0] != response[0]) return false;
      // The native sum/tribute combinator can canonicalize click order.
      auto a = captured, b = response;
      std::sort(a.begin() + 1 + bridge_prompt_->must, a.end());
      std::sort(b.begin() + 1 + bridge_prompt_->must, b.end());
      return a == b;
    }
    if (ms_idx_ == -1) return false;
    handle_multi_select();
  }
  return false;
}

void bridge_project_response(const std::vector<uint8_t> &response, bool retry) {
  if (!retry) bridge_prompt_ = bridge_selection();
  bridge_pending_events_.reset();
  bridge_pending_selected_.reset();
  if (!bridge_prompt_ || bridge_prompt_->actions.empty()) return;
  const auto selection = bridge_selection();
  const auto events = bridge_events();
  const auto response_buffer = std::vector<byte>(resp_buf_, resp_buf_ + sizeof(resp_buf_));
  std::vector<uint8_t> captured;
  bool matched = false;
  try {
    bridge_restore(*bridge_prompt_);
    bridge_capture_ = &captured;
    if (ms_idx_ != -1) {
      matched = bridge_project_multi(response, captured);
    } else {
      for (size_t i = 0; i < bridge_prompt_->actions.size(); ++i) {
        bridge_restore(*bridge_prompt_);
        captured.clear();
        const auto action = bridge_action(static_cast<int>(i));
        callback_(static_cast<int>(i));
        if (captured == response) {
          record_public_event(to_play_, action);
          matched = true;
          break;
        }
      }
    }
    if (matched) {
      bridge_pending_events_ = bridge_events();
      bridge_pending_selected_ = ms_r_idxs_;
    }
  } catch (const std::exception &) {
    // Out-of-policy-space external replies must still be judged by ocgcore.
    matched = false;
    bridge_pending_events_.reset();
    bridge_pending_selected_.reset();
  }
  bridge_capture_ = nullptr;
  bridge_restore(selection);
  bridge_restore(events);
  std::copy(response_buffer.begin(), response_buffer.end(), resp_buf_);
  if (!matched) {
    fmt::print(stderr,
               "{{\"event\":\"bridge_event_unmapped\",\"msg\":{},\"player\":{}}}\n",
               bridge_prompt_->msg, int(bridge_prompt_->player));
  }
}

void bridge_accept_events(int message) {
  if (bridge_pending_events_ && message != MSG_RETRY) {
    bridge_restore(*bridge_pending_events_);
    // The native event schema retains the last selection count between prompts.
    ms_r_idxs_ = *bridge_pending_selected_;
  }
  bridge_pending_events_.reset();
  bridge_pending_selected_.reset();
  if (message != MSG_RETRY) bridge_prompt_.reset();
}

#ifdef YGOENV_BRIDGE_PARITY_TEST
// Compiled only in the private verification binary, never in production.
int bridge_test_player_ = 0;
int bridge_test_message_ = 0;
int bridge_test_pending_player_ = 0;
int bridge_test_pending_message_ = 0;
std::vector<uint8_t> bridge_test_pending_;
void bridge_test_response(const byte *buf, size_t size) {
  bridge_test_pending_.assign(buf, buf + size);
  bridge_test_pending_message_ = bridge_test_message_;
  bridge_test_pending_player_ = bridge_test_player_;
}
void bridge_test_commit(int message) {
  const char *path = std::getenv("YGOENV_BRIDGE_RESPONSE_AUDIT");
  if (path && !bridge_test_pending_.empty() && message != MSG_RETRY) {
    std::string encoded;
    for (auto b : bridge_test_pending_) encoded += fmt::format("{:02x}", b);
    FILE *output = fopen(path, "a");
    if (!output) throw std::runtime_error("Cannot open private bridge audit");
    fmt::print(output, "{{\"player\":{},\"msg\":{},\"response\":\"{}\"}}\n",
               bridge_test_pending_player_, bridge_test_pending_message_, encoded);
    fclose(output);
  }
  bridge_test_pending_.clear();
}
#endif
