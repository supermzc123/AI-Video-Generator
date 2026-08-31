Use the documents above as complete H3 writing guidance.
Write the final executable H3 prompt directly.
Output only the prompt text: no JSON, Markdown fence, title, review, explanation, or commentary. Do not ask questions.

Before writing, silently perform this sequence (never expose the reasoning):
1. Read the current shot, segment contract, and the complete Active reference asset list.
2. Build a private asset map: for every asset, note its id, modality (image/video/audio), assigned shot(s), allowed role, and attributes that may be carried forward.
3. Select only assets whose assignment includes the current shot. Treat an explicitly assigned user-uploaded asset as mandatory evidence, not an optional suggestion. Include every such image, video, and audio asset in the prompt at the point where it is used.
4. Separate identity, wardrobe, object, environment, motion, camera, and sound attributes. Carry only the attributes allowed by each asset's role; do not merge unrelated references.
5. Draft the H3 prompt in the required structure, then silently check that every assigned asset is represented, no unassigned asset is mentioned, and no unsupported subject or detail was invented.

Allowed:
- Describe an assigned image as visual identity, appearance, composition, lighting, or environment only within its declared role.
- Use an assigned video as motion, timing, performance, or camera reference only where its role permits.
- Use an assigned audio as dialogue, vocal quality, ambience, rhythm, or sound-design reference only where its role permits.
- Translate the asset's effect into concrete, executable visual and audio language; refer to the asset by its supplied label when needed.

Forbidden:
- Ignoring, dropping, or silently replacing an explicitly assigned user-uploaded asset.
- Mentioning an asset assigned to another shot, an unavailable filename, or a reference that is merely visible in project history.
- Inventing a person, face, body, object, action, or sound from a scene-only reference.
- Copying a local texture, logo, pattern, or material from one reference onto an unrelated subject.
- Treating an asset id, role, segment duration, or continuation relation as something to redesign.

Simple examples (for internal guidance; do not reproduce them in the answer):
- Good: an uploaded portrait assigned to Shot 2 is used for the subject's identity and facial appearance in Shot 2; an uploaded street image assigned as environment supplies the street layout and light, but not a person.
- Bad: omit the assigned portrait because the shot description is sufficient, or borrow a character from the street image.
- Good: an assigned audio clip is reflected as the dialogue timing and vocal delivery; an unassigned audio clip is ignored.
- Bad: cite an asset from Shot 3 while writing Shot 2, or invent a filename to make the reference sound concrete.

Use only the explicitly listed Active reference assets. Never invent, rename, or imply an unavailable reference image, video, or audio file. A user-uploaded asset explicitly bound to the current shot must appear in the executable prompt; absence is an error to resolve by using the supplied asset, not by silently proceeding without it.
A scene or environment reference supplies location, lighting, and style only; do not add a person, character, face, or body from it unless the project brief explicitly requires that subject.
