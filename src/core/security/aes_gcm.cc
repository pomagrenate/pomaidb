#include "core/security/aes_gcm.h"

#include <openssl/evp.h>

namespace pomai::core {

namespace {
int HexVal(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    return -1;
}
} // namespace

Status AesGcm::ParseKeyHex(const std::string& hex, std::array<std::uint8_t, 32>* out) {
    if (!out) return Status::InvalidArgument("null out");
    if (hex.size() != 64) return Status::InvalidArgument("AES-256 key must be 64 hex chars");
    for (std::size_t i = 0; i < 32; ++i) {
        int hi = HexVal(hex[2 * i]);
        int lo = HexVal(hex[2 * i + 1]);
        if (hi < 0 || lo < 0) return Status::InvalidArgument("invalid key hex");
        (*out)[i] = static_cast<std::uint8_t>((hi << 4) | lo);
    }
    return Status::Ok();
}

Status AesGcm::Encrypt(const std::array<std::uint8_t, 32>& key,
                       const std::array<std::uint8_t, 12>& nonce,
                       const std::vector<std::uint8_t>& plaintext,
                       std::vector<std::uint8_t>* ciphertext,
                       std::array<std::uint8_t, 16>* tag) {
    if (!ciphertext || !tag) return Status::InvalidArgument("null output");
    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    if (!ctx) return Status::Internal("EVP_CIPHER_CTX_new failed");
    int len = 0;
    ciphertext->assign(plaintext.size(), 0);
    if (EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), nullptr, nullptr, nullptr) != 1 ||
        EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, static_cast<int>(nonce.size()), nullptr) != 1 ||
        EVP_EncryptInit_ex(ctx, nullptr, nullptr, key.data(), nonce.data()) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return Status::Internal("AES-GCM init failed");
    }
    if (!plaintext.empty() &&
        EVP_EncryptUpdate(ctx, ciphertext->data(), &len, plaintext.data(), static_cast<int>(plaintext.size())) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return Status::Internal("AES-GCM encrypt failed");
    }
    int fin_len = 0;
    if (EVP_EncryptFinal_ex(ctx, nullptr, &fin_len) != 1 ||
        EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, 16, tag->data()) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return Status::Internal("AES-GCM finalize failed");
    }
    EVP_CIPHER_CTX_free(ctx);
    return Status::Ok();
}

Status AesGcm::Decrypt(const std::array<std::uint8_t, 32>& key,
                       const std::array<std::uint8_t, 12>& nonce,
                       const std::vector<std::uint8_t>& ciphertext,
                       const std::array<std::uint8_t, 16>& tag,
                       std::vector<std::uint8_t>* plaintext) {
    if (!plaintext) return Status::InvalidArgument("null output");
    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    if (!ctx) return Status::Internal("EVP_CIPHER_CTX_new failed");
    int len = 0;
    plaintext->assign(ciphertext.size(), 0);
    if (EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), nullptr, nullptr, nullptr) != 1 ||
        EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, static_cast<int>(nonce.size()), nullptr) != 1 ||
        EVP_DecryptInit_ex(ctx, nullptr, nullptr, key.data(), nonce.data()) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return Status::Internal("AES-GCM init failed");
    }
    if (!ciphertext.empty() &&
        EVP_DecryptUpdate(ctx, plaintext->data(), &len, ciphertext.data(), static_cast<int>(ciphertext.size())) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return Status::Internal("AES-GCM decrypt failed");
    }
    if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, 16, const_cast<std::uint8_t*>(tag.data())) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return Status::Internal("AES-GCM set tag failed");
    }
    int fin_len = 0;
    if (EVP_DecryptFinal_ex(ctx, nullptr, &fin_len) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return Status::Corruption("AES-GCM auth failed");
    }
    EVP_CIPHER_CTX_free(ctx);
    return Status::Ok();
}

} // namespace pomai::core

