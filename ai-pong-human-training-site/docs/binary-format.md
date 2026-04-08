# AI Pong Binary Trajectory Format

The browser stores training chunks as fixed-width little-endian binary files with the
extension `.aipong`.

The goal is to keep human gameplay trajectories compact while still preserving enough
context for supervised learning and later analysis.

## Header

Each file starts with a 64-byte header.

| Offset | Type | Field |
| --- | --- | --- |
| 0 | 8 bytes | Magic string: `AIPONG1\0` |
| 8 | uint16 | Format version, currently `1` |
| 10 | uint16 | Header bytes, currently `64` |
| 12 | uint16 | Record bytes, currently `48` |
| 14 | uint16 | Sample rate, currently `10` Hz |
| 16 | uint32 | Sample count |
| 20 | uint32 | Bot checkpoint update |
| 24 | uint32 | Chunk duration in milliseconds |
| 28 | uint16 | Final left/human score at flush time |
| 30 | uint16 | Final right/bot score at flush time |
| 32 | uint16 | Human paddle hits in this chunk |
| 34 | uint16 | Bot paddle hits in this chunk |
| 36 | uint16 | Max rally length in this chunk |
| 38 | uint16 | Human input action changes |
| 40 | uint16 | Visible-tab samples |
| 42 | uint16 | Chunk flags, bit 0 means match ended |
| 44 | uint32 | Chunk index within browser match id |
| 48 | 16 bytes | Reserved |

## Record

Each sample record is 48 bytes.

| Offset | Type | Field |
| --- | --- | --- |
| 0 | int16[9] | Human-side observation, quantized by `value / 32767` |
| 18 | int16[9] | Bot-side observation, quantized by `value / 32767` |
| 36 | uint8 | Human action: `0` stay, `1` up, `2` down |
| 37 | uint8 | Bot action: `0` stay, `1` up, `2` down |
| 38 | int16 | Human reward, quantized by `value / 1000` |
| 40 | int16 | Bot reward, quantized by `value / 1000` |
| 42 | uint8 | Step flags |
| 43 | uint8 | Human score |
| 44 | uint8 | Bot score |
| 45 | uint16 | Rally hits at sample time |
| 47 | uint8 | `1` if browser tab was visible |

Step flags:

| Bit | Meaning |
| --- | --- |
| 0 | Match ended |
| 1 | Human paddle hit |
| 2 | Bot paddle hit |
| 3 | Bot scored |
| 4 | Human scored |

## Supervised Training Target

For the simplest human-imitation model:

- input: `humanObservation`
- label: `humanAction`

The file also stores `botObservation`, `botAction`, and rewards so you can later filter
or weight samples by how the current bot performed against humans.
