import discord_bridge as db


def test_discord_image_attachment_limits_are_hard_capped():
    assert db._resolve_attachment_limit({"image_attachment_max_mb": 1000}) == db.DEFAULT_IMAGE_ATTACHMENT_MAX_BYTES
    assert db._resolve_attachment_count({"max_image_attachments": 1000}) == db.DEFAULT_IMAGE_ATTACHMENT_MAX_COUNT


def test_discord_image_signatures_must_match_declared_format():
    assert db._image_signature_matches(b"\x89PNG\r\n\x1a\nrest", ".png")
    assert not db._image_signature_matches(b"<script>bad</script>", ".png")
    assert db._image_signature_matches(b"RIFF1234WEBPrest", ".webp")
    assert not db._image_signature_matches(b"RIFF1234NOPErest", ".webp")
