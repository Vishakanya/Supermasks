# 8 categories × 40 images × 6 blocks = 1920 trials
# Each image appears exactly once per block, either with a different mask or unmasked
# Block order is randomized for each subject
# Mondrian and Gaussian masks are generated for every trial
# Preloading of images and masks to prevent frame drops:
# Adversarial, suppressive, and phase-scrambled masks—each image has its own specific mask, preloaded via _preload_images_and_masks.
# Hybrid logic: image and mask durations are calculated in seconds and then converted into frames
# Automated detection of 120Hz monitors; defaults to 60Hz if not detected

from pathlib import Path
import csv
import random
import numpy as np
from PIL import Image
from psychopy import visual, event, core

# =========================
# PATHS & PARAMETERS
# =========================
CODE_ROOT = Path(__file__).parent
DATA_DIR = CODE_ROOT / 'data'

STIM_LABELS = ['cup', 'bag', 'airplane', 'truck', 'dog', 'cat', 'man', 'woman']
N_IMAGES_PER_CATEGORY = 40
BLOCK_TYPES = ['adversarial','unmasked','gaussian','mondrian','phase-scrambled','suppressive']  # 6 blocks

# FRAME_RATE = 60.0  # change to 60 or 120Hz
# FRAME_DURATION = 1.0 / FRAME_RATE   # we use a hybrid logic self.FRAME_RATE and FRAME_DURATION
# FIXED_SOA_FRAMES = 0

# Mondrian generation parameters
MONDRIAN_N_SHAPES = 150
MONDRIAN_MIN_SIZE = 15
MONDRIAN_MAX_SIZE = 50
GAUSSIAN_SIGMA = 10

# =========================
# MONDRIAN & GAUSSIAN MASKS GENERATORS
# =========================
def generate_mondrian_mask(width, height):
    width = int(width)
    height = int(height)
    mask = np.ones((height, width, 3)) * 0.5  # starts at an average luminance of 0.5
    for _ in range(MONDRIAN_N_SHAPES):
        w = np.random.randint(MONDRIAN_MIN_SIZE, MONDRIAN_MAX_SIZE)
        h = np.random.randint(MONDRIAN_MIN_SIZE, MONDRIAN_MAX_SIZE)
        x = np.random.randint(0, max(1, width - w))
        y = np.random.randint(0, max(1, height - h))
        color = np.random.rand(3)             # then colors uniformly distributed between 0 and 1
        mask[y:y+h, x:x+w, :] = color
    return np.clip(mask, 0, 1)
    
def generate_gaussian_mask(width, height):
    noise = np.random.normal(loc=0.5, scale=0.2, size=(int(height), int(width), 3))
    noise = np.clip(noise, 0, 1)
    return noise

# =========================
# EXPERIMENT CLASS
# =========================
class ExperimentPlayer:
    def __init__(self, subject_id):
        self.subject_id = subject_id
        
        # --- Prepare window & stimuli ---
        # Native fullscreen at native resolution, pixel units for true aspect
        self.win = visual.Window(fullscr=True, units='pix', color='gray')
        # Allow pressing Esc to exit
        event.globalKeys.clear()
        event.globalKeys.add(key='escape', func=self._exit)
        
        # === FRAMERATE ===
        self.FRAME_RATE = self.win.getActualFrameRate(nIdentical=20, nMaxFrames=200) # verification of the detected frame rate

        if self.FRAME_RATE is None:
            self.FRAME_RATE = 60.0
            status = "FALLBACK"
        else:
            status = "DETECTED"

        self.FRAME_DURATION = 1.0 / self.FRAME_RATE
        
        debug_msg = f"{status}: {self.FRAME_RATE:.1f} Hz"

        debug_text = visual.TextStim(self.win, text=debug_msg, color='white')
        debug_text.draw()
        self.win.flip()
        core.wait(2)

        # === DURATIONS ===
        self.STIM_DURATION_SEC = 1 / self.FRAME_RATE    # CHOOSE IMAGE DURATION 💖 
        self.MASK_DURATION_SEC = 18 / self.FRAME_RATE   # CHOOSE MASK DURATION 💖  # 18 for 0.15s at 120Hz
        self.MASK_SOA_SEC = 0.0                         # SOA

        # Conversion into frames (hybrid logic)
        self.STIM_DURATION_FRAMES = int(self.STIM_DURATION_SEC * self.FRAME_RATE)
        self.MASK_DURATION_FRAMES = int(self.MASK_DURATION_SEC * self.FRAME_RATE)
        self.MASK_SOA_FRAMES = int(self.MASK_SOA_SEC * self.FRAME_RATE)

        # compute sizes based on window resolution
        width, height = self.win.size
        # fixation dot: 2% of screen height, in pixel units
        self.fixation = visual.Circle(self.win, radius=height*0.005, fillColor='white', lineColor='white')

        # positions 25% out from center
        mult = 2
        self.SLOT_POSITIONS = {
            'up': (0, height*0.025*mult),
            'left': (-width*0.025*mult, 0),
            'right': (width*0.025*mult, 0),
            'down': (0, -height*0.025*mult),
            'up_left': (-width*0.025*mult, height*0.025*mult),
            'up_right': (width*0.025*mult, height*0.025*mult),
            'down_left': (-width*0.025*mult, -height*0.025*mult),
            'down_right': (width*0.025*mult, -height*0.025*mult),
        }
        # rectangle size for highlighting choice
        self.rect_size = (width * 0.03 * mult, height * 0.015 * mult)

        # Create one TextStim per label, pixel units
        self.options = {label: visual.TextStim(self.win, text=label, color='white', units='pix') for label in STIM_LABELS}
        # Visibility prompt & break text
        self.vis_prompt = visual.TextStim(self.win, text="Visibility?", pos=(0,0), color='white', units='pix')
        self.break_text = visual.TextStim(self.win, text="Break time! Press SPACE to continue.", color='white', units='pix')
        
        # RNG for jitter & randomization
        self.rng = random.Random(subject_id)
        # load trial structure & results buffer
        self.trials = self._load_trials_from_csv()
        self.results = []

        # Preloads all file-based images and masks
        self._preload_images_and_masks()

    # Generate structure trials within-image design
    def _generate_trials(self):
        trials = []
        for cat in STIM_LABELS:
            for img_idx in range(1, N_IMAGES_PER_CATEGORY+1):
                image_name = f"{cat}_{img_idx:02d}"
                for block_type in BLOCK_TYPES:
                    trials.append({
                        'stim_label': cat,
                        'image_name': image_name,
                        'mask_type': block_type
                    })

        # Randomizes blocks order per subject
        block_order = self.rng.sample(BLOCK_TYPES, len(BLOCK_TYPES))
        randomized_trials = []
        for block_type in block_order:
            block_trials = [t for t in trials if t['mask_type']==block_type]
            self.rng.shuffle(block_trials)
            randomized_trials.extend(block_trials)

        # Adding block and trial number
        for i, t in enumerate(randomized_trials):
            t['block'] = BLOCK_TYPES.index(t['mask_type'])+1
            t['trial'] = i+1
            t['stim_duration_frames'] = self.STIM_DURATION_FRAMES
            t['mask_duration_frames'] = self.MASK_DURATION_FRAMES
            t['mask_soa_frame'] = self.MASK_SOA_FRAMES

        return randomized_trials

    # Loads trials from CSV created by generator.py
    def _load_trials_from_csv(self):
        csv_path = DATA_DIR / f"sub_{self.subject_id}" / f"sub_{self.subject_id}_trial_structure.csv"

        trials = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Injects pre-converted frame-based durations
                row['stim_duration_frames'] = self.STIM_DURATION_FRAMES     # Stimulus display duration 💖 
                row['mask_duration_frames'] = self.MASK_DURATION_FRAMES     # Stimulus display duration 💖 
                row['mask_soa_frame'] = self.MASK_SOA_FRAMES                # SOA
                row['adversarial_label'] = row.get('adversarial_label', '') # security for adversarial labels
                trials.append(row)
        return trials

    # -------------------------
    # Preload specific images and masks
    # -------------------------
    def _preload_images_and_masks(self):
        for t in self.trials:
            # Image
            stim_path = Path(t['stim_path'])
            t['stim_img'] = visual.ImageStim(self.win, image=str(stim_path))
            #print("Image path:", stim_path)  # print to verify paths for image stimuli

            # File-based mask if applicable, directly from CSV
            mask_type = t['mask_type']
            if mask_type in ['suppressive','adversarial','phase-scrambled']:
                mask_path = Path(t['mask_path'])        # path from CSV
                #print("Mask path loaded:", mask_path)  # print to verify paths for adversarial, suppressive, and phase-scrambled masks

                t['mask_stim'] = visual.ImageStim(self.win, image=str(mask_path), colorSpace='rgb')
            else:
                t['mask_stim'] = None  # mondrian/gaussian/unmasked generated on the fly
    
    # -------------------------
    # Run experiment
    # -------------------------
    def run(self):
        blocks = sorted({t['block'] for t in self.trials})
        for b in blocks:
            block_trials = [t for t in self.trials if t['block']==b]
            for t in block_trials:
                self._run_trial(t)
            if b == 3:
                self._show_break() 
        self._save_results()
        self.win.close()

    # -------------------------
    # Run single trial
    # -------------------------
    def _run_trial(self, t):
        self.fixation.draw()
        self.win.flip()
        core.wait(self.rng.uniform(0.3,0.5)) # randomized duration for fixation dot
        
        # Stimulus
        stim = t['stim_img']
        stim.size = [dim * 2 for dim in stim.size] # double display size (128x128)
        stim.draw()
        self.win.flip()
        print(f"[DEBUG] Stimulus {t['stim_label']}: {self.STIM_DURATION_FRAMES} frames (~{self.STIM_DURATION_FRAMES*self.FRAME_DURATION:.3f} s)") # display stimulus duration
        core.wait(self.STIM_DURATION_FRAMES * self.FRAME_DURATION) # stimulus duration hybrid logic

        # Fixed SOA
        print(f"[DEBUG] SOA: {self.MASK_SOA_FRAMES} frames (~{self.MASK_SOA_FRAMES*self.FRAME_DURATION:.3f} s)") # display SOA duration
        core.wait(self.MASK_SOA_FRAMES * self.FRAME_DURATION) # SOA duration hybrid logic

        # Mask
        mask_type = t['mask_type']
        if mask_type == 'mondrian':
            mask_array = generate_mondrian_mask(*stim.size)  # same displayed surface area, same number of pixels (check settings for comparison with other masks)
            mask_img = Image.fromarray((mask_array*255).astype(np.uint8))
            mask = visual.ImageStim(self.win, image=mask_img, colorSpace='rgb')  # all masks go through the same display pipeline, even phase-scrambled
        elif mask_type == 'gaussian':
            mask_array = generate_gaussian_mask(*stim.size)
            mask_img = Image.fromarray((mask_array*255).astype(np.uint8))
            mask = visual.ImageStim(self.win, image=mask_img, colorSpace='rgb')
        elif mask_type == 'unmasked':
            mask = None
        else:
            mask = t['mask_stim']  # mask file-based

        if mask is not None:
            mask.size = stim.size
            mask.draw()
            self.win.flip()
            print(f"[DEBUG] Mask {mask_type}: {self.MASK_DURATION_FRAMES} frames (~{self.MASK_DURATION_FRAMES*self.FRAME_DURATION:.3f} s)") # display mask duration
            core.wait(self.MASK_DURATION_FRAMES * self.FRAME_DURATION) # mask duration hybrid logic
        else:
            self.win.color = 'gray' # add a gray screen in the unmasked condition to balance the total duration
            self.win.flip()
            core.wait(self.MASK_DURATION_FRAMES * self.FRAME_DURATION)

        # randomized 8AFC response
        slots = list(self.SLOT_POSITIONS.keys())[:len(STIM_LABELS)]
        self.rng.shuffle(slots)

        label_to_slot = dict(zip(STIM_LABELS, slots))

        # numeric keypad -> position mapping
        KEY_TO_SLOT = {
            'num_7':'up_left',
            'num_8':'up',
            'num_9':'up_right',
            'num_4':'left',
            'num_6':'right',
            'num_1':'down_left',
            'num_2':'down',
            'num_3':'down_right'
        }

        # position -> label mapping
        slot_to_label = {slot: lbl for lbl, slot in label_to_slot.items()}

        # draw options & fixation
        for label, txt in self.options.items():
            txt.setPos(self.SLOT_POSITIONS[label_to_slot[label]])
            txt.draw()
        self.fixation.draw()
        self.win.flip()

        resp_clock = core.Clock()
        event.clearEvents()
        key, rt = event.waitKeys(
            keyList=list(KEY_TO_SLOT.keys()) + ['escape'],
            timeStamped=resp_clock
        )[0]
        if key == 'escape':
            self._exit()
        slot = KEY_TO_SLOT[key]
        choice = slot_to_label[slot]
        correct = (choice == t['stim_label'])
        
        # highlight selected choice
        rect = visual.Rect(
            self.win,
            width=self.rect_size[0],
            height=self.rect_size[1],
            pos=self.SLOT_POSITIONS[label_to_slot[choice]],
            lineColor='yellow',
            fillColor=None,  # <-- prevent white fill
            lineWidth=3,
            units='pix'
        )
        rect.draw()
        for txt in self.options.values():
            txt.draw()
        self.win.flip()
        core.wait(0.2)
        
        # # feedback
        # color = 'green' if correct else 'red'
        # self.fixation.setFillColor(color)
        # self.fixation.draw()
        # self.win.flip()
        # core.wait(FEEDBACK_DURATION)

        # visibility rating
        self.vis_prompt.draw()
        self.win.flip()
        vis_clock = core.Clock()
        event.clearEvents()
        vk, vrt = event.waitKeys(keyList=['1','2','3','4','escape'], timeStamped=vis_clock)[0]
        if vk=='escape': self._exit()

        # save data
        self.results.append({
            **t,
            'response_key': key,
            'response_label': choice,
            'correct': int(correct),
            'response_rt': rt,
            'visibility_rating': int(vk),
            'visibility_rt': vrt
        })

    # break screen
    def _show_break(self):
        break_text = visual.TextStim(
            self.win,
            text="BREAK TIME !\n\nThe experiment will resume automatically in 7 mn\n\nor press ENTER when ready",
            color='white',
            units='pix',
            height=30,
            wrapWidth=1000
        )

        countdown = visual.TextStim(
            self.win,
            text="07:00",
            pos=(0, -150),
            color='yellow',
            units='pix',
            height=40
        )

        timer = core.Clock()

        while timer.getTime() < 420:  # 7 minutes break
            remaining = 420 - timer.getTime()
            minutes = int(remaining // 60)
            seconds = int(remaining % 60)
            countdown.text = f"{minutes:02d}:{seconds:02d}"

            break_text.draw()
            countdown.draw()
            self.win.flip()

            keys = event.getKeys(['return', 'escape'])
            if 'return' in keys:
                break
            if 'escape' in keys:
                self._exit()

    # -------------------------
    # Save results
    # -------------------------
    def _save_results(self):
        if len(self.results)==0: return
        outdir = DATA_DIR / f"sub_{self.subject_id}"
        outdir.mkdir(exist_ok=True)
        outfile = outdir / f"sub_{self.subject_id}_results.csv"
        with open(outfile,'w',newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.results[0].keys()))
            writer.writeheader()
            for row in self.results:
                writer.writerow(row)
        print("Results saved:", outfile)

    # -------------------------
    # Exit
    # -------------------------
    def _exit(self):
        try:
            if len(self.results) > 0:
                self._save_results()
        except:
            pass
        core.quit()

# =========================
# RUN
# =========================
if __name__=='__main__':
    # Hard-code a subject ID
    subject_id = 1  # ← Change this as needed
    player = ExperimentPlayer(subject_id)
    player.run()