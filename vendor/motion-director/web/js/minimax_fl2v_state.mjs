export function resolveFl2vEndpointState(shot = {}) {
    const startFile = String(shot?.startImage?.imageFile || "").trim();
    const endFile = String(shot?.endImage?.imageFile || "").trim();
    const hasStart = !!startFile;
    const hasEnd = !!endFile;
    return {
        hasStart,
        hasEnd,
        endOnly: hasEnd && !hasStart,
        startFile,
        endFile,
        badgeKey: hasStart && hasEnd
            ? "fl2v.badge.startEnd"
            : hasEnd
                ? "fl2v.badge.endOnly"
                : "fl2v.badge.i2v",
    };
}
