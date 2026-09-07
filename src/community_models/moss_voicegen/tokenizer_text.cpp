#include "engine/community_models/moss_voicegen/tokenizer_text.h"

#include "engine/framework/tokenizers/llama_bpe.h"
#include "engine/models/moss/shared/delay_prompt.h"

#include <stdexcept>
#include <utility>

namespace engine::models::moss_voicegen {

struct MossVoiceGenTextProcessor::Impl {
    std::shared_ptr<const MossVoiceGenAssets> assets;
    std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> tokenizer;
    int32_t im_start_token_id = 0;
    int32_t im_end_token_id = 0;
};

MossVoiceGenTextProcessor::MossVoiceGenTextProcessor(std::shared_ptr<const MossVoiceGenAssets> assets)
    : impl_(std::make_unique<Impl>()) {
    if (assets == nullptr) {
        throw std::runtime_error("MOSS-VoiceGenerator text tokenizer requires assets");
    }
    engine::tokenizers::LlamaBpeTokenizerSpec spec;
    // The checkpoint ships tokenizer.json and merges.txt but no vocab.json. The framework
    // treats vocab/merges as a pair, so leave both unset and take the vocabulary and the
    // merge ranks from tokenizer.json, which carries them.
    spec.tokenizer_config_path = assets->resources.require_file("tokenizer_config");
    spec.tokenizer_json_path = assets->resources.require_file("tokenizer_json");
    spec.pre_type = engine::tokenizers::LlamaBpePreTokenizer::Qwen2;
    impl_->tokenizer = engine::tokenizers::load_llama_bpe_tokenizer(spec);

    // The config does not carry the chat-control ids; the reference processor resolves
    // them through the tokenizer, so do the same rather than hardcoding Qwen's values.
    const auto im_start = impl_->tokenizer->find_token_id("<|im_start|>");
    const auto im_end = impl_->tokenizer->find_token_id("<|im_end|>");
    if (!im_start.has_value() || !im_end.has_value()) {
        throw std::runtime_error("MOSS-VoiceGenerator tokenizer is missing the chat control tokens");
    }
    impl_->im_start_token_id = *im_start;
    impl_->im_end_token_id = *im_end;
    impl_->assets = std::move(assets);
}

MossVoiceGenTextProcessor::~MossVoiceGenTextProcessor() = default;

engine::codecs::MossTokenRows MossVoiceGenTextProcessor::build_generation_prefix(
    const std::string & text,
    const std::optional<std::string> & instruction,
    const std::optional<std::string> & language) const {
    const auto & config = impl_->assets->config;
    engine::codecs::MossTokenRowBuilder builder(config.num_codebooks, static_cast<int32_t>(config.audio_pad_code));

    // Render the whole turn as one string and encode it in a single pass, the way the
    // reference processor does. Encoding the fragments separately would split merges
    // across the seams — the text's trailing "." and the suffix's "\n" are one token in
    // the reference, two if the suffix is encoded on its own.
    delay::PromptFields fields;
    // Voice design has no reference recording: the speaker comes from the instruction, so
    // the reference slot stays "None" and every other control slot follows the default.
    fields.instruction = instruction;
    fields.language = language;
    fields.text = text;
    const std::string prompt = delay::render_turn(fields);
    builder.push_text_tokens(impl_->tokenizer->encode(prompt, true));
    // Unlike moss_tts_local, the delay family does not seed the audio start token: the
    // model emits it itself on the first step, and generate() keys "is this a
    // continuation" off the last text token, so appending it here would be read as a
    // continuation prompt.

    return builder.finish();
}

}  // namespace engine::models::moss_voicegen
