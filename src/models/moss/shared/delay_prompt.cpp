#include "engine/models/moss/shared/delay_prompt.h"

namespace engine::models::moss::delay {
namespace {

// Template fragments copied verbatim from MossTTSDelayProcessor so the encoded prompt
// matches the reference token-for-token.
constexpr const char * kUserRolePrefix = "user\n";
constexpr const char * kUserReferencePrefix = "<user_inst>\n- Reference(s):\n";
constexpr const char * kUserTextSuffix = "\n- Text:\n";
constexpr const char * kUserInstSuffix = "\n</user_inst>";
constexpr const char * kAssistantTurnPrefix = "\n";
constexpr const char * kAssistantRolePrefix = "assistant\n";
constexpr const char * kNoneValue = "None";
constexpr const char * kImStartToken = "<|im_start|>";
constexpr const char * kImEndToken = "<|im_end|>";

}  // namespace

std::string normalize_field(const std::optional<std::string> & value) {
    if (!value.has_value()) {
        return kNoneValue;
    }
    std::string resolved = *value;
    const auto first = resolved.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return kNoneValue;
    }
    const auto last = resolved.find_last_not_of(" \t\r\n");
    resolved = resolved.substr(first, last - first + 1);
    return resolved.empty() ? kNoneValue : resolved;
}

std::string render_turn(const PromptFields & fields) {
    return std::string(kImStartToken) + kUserRolePrefix
        + kUserReferencePrefix + fields.reference
        + "\n- Instruction:\n" + normalize_field(fields.instruction)
        + "\n- Tokens:\n" + normalize_field(fields.tokens)
        + "\n- Quality:\n" + normalize_field(fields.quality)
        + "\n- Sound Event:\n" + normalize_field(fields.sound_event)
        + "\n- Ambient Sound:\n" + normalize_field(fields.ambient_sound)
        + "\n- Language:\n" + normalize_field(fields.language)
        + kUserTextSuffix + fields.text
        + kUserInstSuffix + kImEndToken
        + kAssistantTurnPrefix + kImStartToken + kAssistantRolePrefix;
}

}  // namespace engine::models::moss::delay
