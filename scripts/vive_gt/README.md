# Vive Ground-Truth Logger (real-world dataset mission)

Goal: record the exact (x, y) position of the Android tablet while you walk,
using the fablab's HTC Vive kit as a measurement instrument. One of the two
handheld **controllers** (the wands) is attached to the tablet; the two
**base stations** on the wall watch it; a PC writes its position to a CSV file
100 times per second with millimeter precision.

This folder is self-contained — copy it to the fablab PC (USB stick is fine).

---

## 0. Know the pieces (glossary)

| Part | What it looks like | What it does |
|---|---|---|
| **Base station** (×2) | Black square box, mounted on the wall, front is dark glass | Sweeps the room with invisible laser lines. Just needs power — it sends nothing to the PC. |
| **Controller / wand** (×2) | The stick you hold, with a ring at the top full of small dimples | The dimples are sensors that see the lasers. This is the object whose position gets logged. **We only need one.** |
| **Headset** | The goggles | We never wear it. It must stay connected to the PC because the controllers talk to the PC *through a radio inside the headset*. Leave it sitting on the desk. |
| **Link box** | Small box with cables between headset and PC | Just the headset's connection to the PC. Your colleague has this wired already. |

## 1. Base stations — already done ✓

You mounted them. Each one only needs its **power adapter plugged into a wall
socket** — base stations send nothing to the PC (the lasers themselves are the
signal). Check: each one's front LED should be **white/green** when powered.
If one blinks, tell me the color and we'll debug.

## 2. Plug the kit into the PC

Everything goes through the **link box** (the small flat box with cables).
The Vive Pro link box has two sides:

- one side with **three ports**: DisplayPort, USB, and a round power socket → this side goes to the PC and the wall
- one side with a **single wide port marked with a triangle** → this is where the headset's cable plugs in
- a **power button** on top

Plug in, in this order:

1. **DisplayPort cable**: link box → a **DisplayPort output on the graphics
   card** — the same row of ports where the monitor is plugged in, at the back
   of the PC. NOT the ports higher up near the mouse/keyboard (those are the
   motherboard's and won't work).
2. **USB cable**: link box → a **USB 3.0 port** on the PC (usually blue inside).
3. **Power adapter**: link box → wall socket.
4. **Headset cable** (the single thick cable coming out of the goggles): plug
   its flat connector into the triangle-marked port on the link box, **triangle
   mark facing up**.
5. Press the **power button on the link box** — it lights up blue.
6. Leave the headset on the desk. Done — you never touch it again.

Tips: if the cables are already routed from your colleague's sessions, every
plug only fits its own port — match shapes. If in doubt, take a photo of the
current state before changing anything, and/or ask your colleague to show you
their startup once.

(If your link box instead has **orange ports and an HDMI cable**, it's the
original Vive model: orange side = headset's three connectors, other side =
HDMI + USB + power to the PC, and there is no power button — it's on whenever
powered.)

## 3. Start SteamVR on the fablab PC

Your colleague already installed everything, so this is just:

1. Turn on the PC, open **Steam** (it may auto-start; otherwise Start menu → Steam).
2. In Steam, top menu **Library**, find **SteamVR** in the left list, press **Play** —
   or click the small **VR icon** at the top-right of the Steam window.
3. A **small SteamVR status window** appears with little icons: a headset, two
   controllers, two base stations. Green = detected and tracking, grey = off,
   blinking green = detected but can't see the lasers.
4. Expected right now: headset icon green (it's plugged in), base stations green,
   controllers grey (they're off).

If SteamVR complains "headset not detected", check the link box cables — or ask
your colleague to show you their normal startup once; whatever they do, do the same.

## 4. Turn on ONE controller

1. Press the small round button **below the trackpad** (the big circular touch
   area). It vibrates and its LED lights up.
2. LED **green** = paired and ready → its icon in the SteamVR window turns green. Done.
3. LED **blue blinking** = not paired to this headset. In the SteamVR status
   window: menu (☰) → **Devices → Pair Controller** and follow the on-screen
   instructions (hold trigger + menu button until it beeps).

Quick check: walk around the room holding the controller. The icon should stay
solid green everywhere in the walking area. If it flickers grey in some corner,
that corner isn't covered — remember it, we'll see it in the data too.

### If icons stay grey ("not connected")

- **Base station icons light up only when a device sees their lasers** — the
  stations have no cable/radio to the PC. A headset lying on the desk facing
  the wall sees nothing, so stations look "not connected" while working fine.
  Hold a powered-on controller up mid-room with clear view of both stations;
  the station icons should light within seconds.
- Check each station's front LED: white/green = working, blue = still starting,
  red/blinking = fault (report it).
- Still grey with a green controller in view → likely both stations on the same
  channel: SteamVR menu (☰) → **Devices → Base Station Settings** → let it
  configure channels automatically.
- Controller LED blinking blue = not paired: SteamVR menu (☰) →
  **Devices → Pair Controller**, then hold trigger + menu button until it beeps.

## 5. Install the logger (one-time, PowerShell on the fablab PC)

Copy this folder to the PC, e.g. to `C:\vive_gt\`. Then open PowerShell
(Start menu, type "powershell") and run:

```powershell
cd C:\vive_gt
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

If `python` is not found, install it first from https://python.org (check
"Add python.exe to PATH" during install), close and reopen PowerShell.

## 6. Record a session

With SteamVR running and the controller icon green:

```powershell
cd C:\vive_gt
.venv\Scripts\python vive_logger.py --out sessions --rate 100
```

Add `--viz` to also open a **live 3D window**: a red dot at the controller's
current position with a green trail fading out over the last 10 seconds
(`--trail 20` for a longer trail). Trigger presses appear as orange ×.
Closing the window ends the session, same as Ctrl+C. Logging runs in its own
thread at full rate — the viz never affects the CSV.

You should see `logging device N: controller LHR-XXXXXXXX` and then a status
line updating every second with the live position in meters. Walk around and
watch x/z change — that's the moment you know the whole chain works.

**Session ritual** (for the real feasibility test, with the tablet also logging):

1. Put the controller down on a table, untouched, for **5 seconds** (this
   measures the jitter).
2. Hold the controller against the tablet and give **3 sharp taps**, pressing
   the **trigger** (the button under your index finger) at each tap. The taps
   show up in the tablet's motion sensors and the trigger presses in the CSV —
   that's how we align the two clocks afterwards.
3. Walk your loop around the room, tablet + controller held together.
4. Repeat the 3 taps at the end.
5. Back at the PC: press **Ctrl+C** in the PowerShell window. It prints a
   summary and the CSV filename.

For a first dry run, skip the ritual: just start the logger, walk a loop with
the controller in hand, Ctrl+C. That already answers the feasibility question.

## 7. Look at the result

```powershell
.venv\Scripts\python plot_session.py sessions\vive_gt_20260722_150000.csv
```

(use the real filename printed at step 6). This prints a quality report and
saves a PNG next to the CSV: your walked path seen from above.

**Pass criteria** — the Vive is good enough for our dataset if:

- valid poses ≥ 95 % of samples, longest dropout < 0.5 s
- no dead zone when your own body is between the controller and one station
- stationary jitter (the 5-second still phase) < 5 mm

Send me the PNG + the printed report and we decide together.

## 8. Attaching the controller to the tablet

The controller's position is only a valid ground truth for the tablet if the
two are **rigidly fixed together** (if the controller wobbles relative to the
tablet, that wobble becomes fake position error in the dataset).

- **For the feasibility test: no mount needed.** Press the controller flat
  against the back of the tablet and hold both with two hands, or strap them
  with rubber bands / velcro / tape. Totally fine for one session.
- **For the real dataset**: we'll 3D-print a proper clamp (the fablab printer)
  so every session has the controller in the exact same place. Design comes
  later — not needed today.

## 9. Give the coordinates a real-world reference (after Room Setup)

SteamVR's **Room Setup** ("Configurer la pièce") calibrates the tracking frame:
floor at height 0, origin and axes placed somewhere in the traced field. Two rules:

- **Do Room Setup once, then never again** during the collection campaign, and
  don't move the base stations. Re-running it (or moving a station) changes the
  frame — sessions before and after would live in different coordinates.
- The origin/axes are still arbitrary w.r.t. the physical room. The anchor
  protocol below pins them to real room coordinates — and doubles as a
  start-of-day check that nothing moved.

**One-time setup:**

1. Choose a room origin (e.g. the SW corner) and two axis directions along the walls.
2. Tape **4 marks** on the floor, spread out, NOT in a straight line.
3. Measure each mark's (x, y) from your origin with a tape measure (±1 cm is fine).
4. Copy `anchors.example.json` to `anchors.json` and put your measurements in it.

**Each collection day (2 minutes):**

1. Start the logger for a short dedicated session.
2. Place the controller on each anchor **in the order listed in anchors.json**,
   hold it still, and **hold the trigger ~1 second** on each.
3. Ctrl+C, then:

```powershell
.venv\Scripts\python register_frame.py sessions\vive_gt_XXXXXX.csv anchors.json
```

It prints per-anchor residuals and saves `frame_calibration.json`. Residuals of
a few mm–cm = healthy; a warning above 5 cm = redo (wrong order, wrong
measurement, or a station moved).

**To convert any session into room coordinates** (adds `room_x`/`room_y` columns):

```powershell
.venv\Scripts\python register_frame.py --apply sessions\vive_gt_YYYYYY.csv
```

Keep using the raw CSVs as the primary record; the calibration can always be
re-applied later.

---

## Appendix: coordinates and advanced setup

- OpenVR is **y-up**: the floor plane is (x, z), and `plot_session.py` plots
  x vs z. Height is y.
- The map origin is wherever your colleague's "room setup" put it. That's fine
  for feasibility. For the real dataset we'll register the Vive frame to the
  room plan by holding the controller at 3+ known points (trigger-marking each).
- **If the headset ever can't be connected** (different PC, no DisplayPort):
  SteamVR can run without it ("null driver": in
  `Steam\steamapps\common\SteamVR\resources\settings\default.vrsettings` set
  `"requireHmd": false`, `"forcedDriver": "null"`, `"activateMultipleDrivers": true`,
  and `"enable": true` under `"driver_null"`) — but then the controller needs a
  separate USB dongle, since its radio receiver lives in the headset. Avoid
  unless forced. Reference:
  https://roadtovr.com/how-to-use-the-htc-vive-tracker-without-a-vive-headset/
