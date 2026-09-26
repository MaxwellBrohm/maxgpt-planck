"""E004 eval fillers (notes (b)). EVAL_FILLERS_NEW: how-to / why Q&A, added to E001's filler pool to make the
default eval fillers (pools_eval.eval_fillers()). H7_FILLERS: the held-out genre for H7 (newfill): personal
small talk (the user shares a feeling or a small story, the assistant responds) and tiny creative tasks (a
two-line poem, a riddle). No value of any pool, no names, no digits, no marker words; 0 word 5-grams shared with
training text (checked in checks_eval.py and purity_e004.py). The item builder drops, per item, every filler
that contains a word of that item's object phrases, head nouns or alias."""

EVAL_FILLERS_NEW = [
    ("Is it bad to crack my knuckles?", "It is mostly harmless: the sound is just gas bubbles collapsing in the joint fluid."),
    ("Why does bread go hard in the fridge?", "Cold air speeds up how the starch recrystallizes, so it firms up faster than at room temperature."),
    ("How can I fall asleep faster?", "Keep the room cool and dark, and get up briefly if you lie awake for too long."),
    ("Why does pasta water need salt?", "Salt seasons the pasta from the inside as it absorbs water in the pot."),
    ("Why do windows fog up in the morning?", "Warm, moist indoor air meets cold glass and the water vapor condenses on it."),
    ("Why do boats float?", "They push aside a weight of water greater than their own, and that water pushes back up."),
    ("How should I store fresh berries?", "Keep them dry and unwashed in the fridge, and rinse them just before eating."),
    ("Why do some cheeses have holes?", "Bacteria in the curd release gas bubbles that get trapped as the cheese ages."),
    ("How can I make houseplants grow fuller?", "Pinch off the growing tips so the plant branches out, and turn the pot now and then."),
    ("What's a good way to learn to juggle?", "Start with one ball tossed hand to hand, then add a second once the rhythm feels easy."),
    ("How do I remove sticker residue?", "Rub a little vegetable oil into it, leave it for a few minutes, then wipe it off with soap."),
    ("Why do cats knead blankets?", "It is a comforting habit from kittenhood that also leaves scent from their paws."),
    ("How can I keep my desk tidy?", "Give every item a home, clear the surface at the end of each day, and keep only what you use daily."),
    ("What makes popcorn pop?", "Water inside each kernel turns to steam and builds pressure until the hull bursts."),
    ("How do I soften hard brown sugar?", "Seal it in a container with a slice of bread overnight and it absorbs the moisture."),
    ("Why do my fingers wrinkle in the bath?", "Nerves tighten the blood vessels under the skin, which pulls the surface into ridges."),
    ("How can I make a small balcony feel cozy?", "Add outdoor cushions, some potted herbs and warm string lights."),
    ("How can I stop a chopping board from slipping?", "Put a damp cloth under it before you start chopping."),
    ("Why does metal feel colder than wood?", "Metal pulls heat away from your skin much faster, even when both are at the same temperature."),
    ("How can I stop my shoelaces coming undone?", "Loop the laces twice before pulling tight, which makes the knot hold much better."),
    ("How can I make my mornings less rushed?", "Lay out clothes the night before, and keep your keys in one spot by the entrance."),
    ("Why do candles flicker?", "Small drafts and uneven melting change how much fuel reaches the flame."),
    ("How do I keep avocados from browning?", "Leave the pit in, squeeze on some lemon juice, and press plastic wrap against the surface."),
    ("What's a good way to warm up before lifting weights?", "Do a few minutes of light cardio, then some easy sets with lighter loads."),
    ("What helps a sunburn heal?", "Cool compresses, aloe, plenty of water and keeping the skin out of the sun while it recovers."),
    ("Why do dogs tilt their heads?", "They may be trying to hear better or to see your face around their muzzle."),
]

H7_FILLERS = [
    # personal small talk
    ("I finally cleaned out my whole closet today.", "That must feel so satisfying, like a fresh start for the season."),
    ("I'm a little nervous about speaking up at work.", "That is really common, and it usually gets easier each time you try."),
    ("My neighbor brought over homemade soup yesterday.", "What a kind gesture, it is lovely when neighbors look out for each other."),
    ("I tried baking bread for the first time and it came out flat.", "First loaves often do, and a warmer spot for rising usually helps a lot."),
    ("I've been feeling a bit tired lately.", "Rest when you can, and be gentle with yourself while your energy comes back."),
    ("I finished a really long novel last night.", "Congratulations, finishing a long read always feels like an accomplishment."),
    ("My plants are finally growing new leaves.", "That is great news, it means they are happy where they are."),
    ("I got lost on my walk this morning but found a nice coffee shop.", "Sometimes the best places turn up when you take a wrong turn."),
    ("I'm proud of myself for going for a run in the rain.", "You should be, that takes real determination."),
    ("I had a really relaxing afternoon.", "I am glad to hear it, quiet afternoons are worth protecting."),
    ("My little brother made me laugh so hard today.", "Moments like that are the best, siblings know exactly how to get us."),
    ("I'm excited, I just started learning to knit.", "How fun, a simple dishcloth is a great first project."),
    ("It has been such a busy week for me.", "That sounds draining, I hope you get some time to recharge soon."),
    ("I saw the most beautiful sunset tonight.", "Sunsets like that make you stop and breathe for a moment."),
    ("I'm in a great mood today.", "That is wonderful, I hope the good feeling lasts all day."),
    ("I finally fixed the wobbly chair myself.", "Nice work, small repairs like that are so satisfying."),
    ("My friend surprised me with flowers.", "What a sweet surprise, flowers brighten any room."),
    # tiny creative tasks
    ("Write a two-line poem about rain.", "Soft rain taps the glass all night, / the whole street hums and dims its light."),
    ("Tell me a riddle.", "What has hands but cannot clap? A clock."),
    ("Write a two-line poem about coffee.", "A warm cup held in both my hands, / the morning slowly understands."),
    ("Give me a short riddle, please.", "What gets wetter the more it dries? A towel."),
    ("Write a tiny poem about the sea.", "Waves fold over, soft and wide, / the shore keeps every secret tide."),
    ("Can you make up a riddle about the moon?", "I change my shape but never leave, and I shine without a light of my own. The moon."),
    ("Write a two-line poem about a cat.", "She naps where sunlight likes to fall, / and owns the house, the chair, the hall."),
    ("Tell me a riddle about walking.", "The more you take, the more you leave behind. Footsteps."),
    ("Write a short poem about snow.", "Quiet flakes on sleeping roofs, / the world goes hushed and soft and smooth."),
    ("Make up a two-line poem about friendship.", "A friend is shelter in the storm, / a steady hand, a place that's warm."),
    ("Tell me a quick riddle for kids.", "What has one eye but cannot see? A needle."),
    ("Write a two-line poem about autumn leaves.", "They drift and spin on the evening air, / and settle softly everywhere."),
    ("Write a two-line poem about the night sky.", "Stars scatter wide like spilled sugar, / the dark holds still so we can look."),
]
