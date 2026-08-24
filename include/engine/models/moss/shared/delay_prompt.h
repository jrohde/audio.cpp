#pragma once

#include <optional>
#include <string>

namespace engine::models::moss::delay {

// The <user_inst> control block every moss_tts_delay checkpoint is trained on. Each model
// fills a different subset: voice design carries the instruction and leaves the reference
// at "None", while the dialogue models do the opposite. Unset fields render as "None",
// which is what the reference processor emits for them.
struct PromptFields {
    // Pre-rendered, because the shape differs per model: a single "None" for voice design,
    // one "[S1]:" line per speaker for dialogue.
    std::string reference = "None";
    std::optional<std::string> instruction;
    std::optional<std::string> tokens;
    std::optional<std::string> quality;
    std::optional<std::string> sound_event;
    std::optional<std::string> ambient_sound;
    std::optional<std::string> language;
    std::string text;
};

// Renders one complete user turn plus the assistant prefix, as a single string ready to be
// encoded in one pass. Encoding the fragments separately would split BPE merges across the
// seams — the text's trailing "." and the following "\n" are one token in the reference,
// two if the suffix is encoded on its own.
std::string render_turn(const PromptFields & fields);

// Trims surrounding whitespace and maps an absent or blank value onto "None".
std::string normalize_field(const std::optional<std::string> & value);

}  // namespace engine::models::moss::delay
