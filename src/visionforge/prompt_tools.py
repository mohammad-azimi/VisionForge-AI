def analyze_prompt(prompt: str) -> dict:
    words = [word for word in prompt.replace(",", " ").split() if word.strip()]
    word_count = len(words)

    checks = {
        "Has output purpose": any(
            term in prompt.lower()
            for term in ["portfolio", "banner", "icon", "linkedin", "hero", "preview"]
        ),
        "Has visual style": any(
            term in prompt.lower()
            for term in ["minimal", "cinematic", "medical", "dashboard", "futuristic", "modern"]
        ),
        "Has color palette": "color palette" in prompt.lower(),
        "Has quality instruction": any(
            term in prompt.lower()
            for term in ["high quality", "sharp", "professional", "clean"]
        ),
        "Avoids text artifacts": "no text" in prompt.lower() and "watermark" in prompt.lower(),
        "Has project context": "project called" in prompt.lower(),
    }

    score = 0
    score += min(word_count, 80) * 0.5

    for is_good in checks.values():
        if is_good:
            score += 10

    score = int(min(score, 100))

    recommendations = []

    if word_count < 30:
        recommendations.append("Add more visual details to guide the model.")
    if not checks["Has color palette"]:
        recommendations.append("Add a clear color palette.")
    if not checks["Has visual style"]:
        recommendations.append("Choose a stronger visual style.")
    if not checks["Avoids text artifacts"]:
        recommendations.append("Add instructions like no text, no logo, no watermark.")
    if score >= 80:
        recommendations.append("Prompt is strong enough for image generation.")

    return {
        "word_count": word_count,
        "score": score,
        "checks": checks,
        "recommendations": recommendations,
    }


def build_seed_plan(base_seed: int, count: int) -> list[int]:
    return [base_seed + index for index in range(count)]
