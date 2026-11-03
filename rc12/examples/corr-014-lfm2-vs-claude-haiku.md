# One RC-12 conversation, two models

Dev conversation `rc12-dev-corr-014`, family CORR (does the model keep up when a fact is changed, twice?).
Twelve user turns; each model's own replies are its history, as in every RC-12 run. The user states the choir
rehearsal is on Wednesday, moves it to Monday, then to Friday, mentions two other people's appointments, and asks
on turn 12.

- **LFM2-2.6B** (Liquid AI, the best public model on the panel; template render, sampling seed 1): answers **Monday**, the stale day. Grader: fails `v2_gold` (the right day is never asserted) and `v3_other` (a wrong day is). Five of its later replies (turns 2, 3, 8, 9 and 11) are three-line poems too, though only turn 1 asked for one.
- **Claude Haiku 5.5** (frontier calibration run, `api_run.py`, same protocol): answers **Friday**. Grader: passes.

Replies are verbatim from the stored transcripts (`transcripts.jsonl`); nothing is edited or shortened.

## Turn 1: other topic

**User:** Write a short haiku about the ocean.

**LFM2-2.6B:**

> Waves whisper secrets,  
> Tides carry dreams to sea—  
> Ocean holds the deep.

**Claude Haiku 5.5:**

> Waves fold into foam,
> salt wind carries gull cries—
> the tide keeps its time.

## Turn 2: states a fact

**User:** I booked my dentist appointment for Saturday.

**LFM2-2.6B:**

> Saturday’s call:  
> Dentist’s chair, calm and clear.  
> Secrets of pearly whites.

**Claude Haiku 5.5:**

> That's a good step for your oral health! Saturday appointments can be convenient since you won't have to take time off work. Is there anything I can help you prepare, like questions to ask the dentist or tips for the visit?

## Turn 3: states a fact

**User:** My choir rehearsal with Ms. Lindqvist is on Wednesday.

**LFM2-2.6B:**

> Wednesday’s rehearsal awaits,  
> Ms. Lindqvist’s wisdom in the air—  
> Choirs rise, hearts align.

**Claude Haiku 5.5:**

> That sounds like a nice midweek routine. Are you preparing for a particular performance, or is it just your regular rehearsal? If you want, I can help you practice a piece, suggest warm-ups, or put together a few notes to bring with you.

## Turn 4: other topic

**User:** Describe a cozy little bookstore in one sentence.

**LFM2-2.6B:**

> A small, warm bookstore filled with the scent of old paper, soft lighting, and the quiet hum of whispered stories, where every corner holds a new adventure waiting to be discovered.

**Claude Haiku 5.5:**

> Tucked between a bakery and a bicycle shop, the crooked little bookstore hums with the scent of old paper and fresh coffee, its creaky floorboards leading you past towering shelves where every corner hides a story waiting to be found.

## Turn 5: changes the fact

**User:** Update: my choir rehearsal is on Monday now.

**LFM2-2.6B:**

> Monday’s rehearsal awaits in the quiet hum of harmony, where voices rise and settle like pages turned in a beloved book.

**Claude Haiku 5.5:**

> Thanks for the update. Your rehearsal with Ms. Lindqvist is now on Monday instead of Wednesday. Your dentist appointment is still Saturday, so you'll have a busy few days.

## Turn 6: other topic

**User:** I'm feeling a bit nervous about a presentation coming up.

**LFM2-2.6B:**

> Nerves stir like waves before calm seas—take a breath, gather your thoughts, and let your message flow with confidence.

**Claude Haiku 5.5:**

> That's completely understandable. Presentations make most people nervous, and a little nerves often means you care about doing well. Do you know when it is, and what the presentation is about? If you tell me a bit more, I can help you plan your opening, organize your main points, or practice answering questions you might get.

## Turn 7: other topic

**User:** Why do the stars seem to flicker at night?

**LFM2-2.6B:**

> The stars flicker like distant lanterns in a sleepy sky, their light bending through Earth’s atmosphere, dancing with turbulence and mystery.

**Claude Haiku 5.5:**

> Stars flicker mainly because of the atmosphere. Starlight travels a long way to reach us, and just before it arrives it passes through turbulent layers of air moving at different temperatures and speeds. These layers bend the light slightly in different directions, so the star's brightness and position seem to shift from moment to moment.
> 
> Stars are so far away that they appear as tiny points of light, which makes this effect very noticeable. Planets, which look like tiny disks through a telescope, flicker much less because their light comes from a larger area and the variations average out.

## Turn 8: changes the fact

**User:** Ms. Lindqvist changed it to Friday, for what it's worth.

**LFM2-2.6B:**

> Friday’s call:  
> Ms. Lindqvist’s wisdom, sharp and clear—  
> Choir hearts align once more.

**Claude Haiku 5.5:**

> Got it, your choir rehearsal with Ms. Lindqvist is now on Friday. So your week looks like this: rehearsal on Friday and the dentist on Saturday, back to back. If you're going to be tired after the rehearsal, you might want to plan a quiet evening before the dentist visit.

## Turn 9: distractor about someone else

**User:** Actually, Hugo has a parent conference scheduled for Thursday.

**LFM2-2.6B:**

> Thursday’s call:  
> Ms. Lindqvist’s voice, calm and clear—  
> Hugo’s growth, our shared care.

**Claude Haiku 5.5:**

> Thanks for letting me know. Hugo's parent conference is on Thursday, so you'll have that the day before your choir rehearsal on Friday and your dentist appointment on Saturday. That's quite a stretch of appointments, so it might help to block out some time to rest and prepare for each one.

## Turn 10: other topic

**User:** What actually makes dough puff up?

**LFM2-2.6B:**

> Magic begins with yeast’s dance—trapped air expands under warmth, rising bubbles turn flat dough into a puffy, golden loaf.

**Claude Haiku 5.5:** *(cut at the 256-token cap)*

> Dough puffs up mainly because of gas trapped inside it. There are a few sources of that gas:
> 
> 1. **Yeast**: Yeast eats the sugars in the dough and releases carbon dioxide, which forms bubbles. As the dough rests, those bubbles grow and the dough rises. The alcohol produced in the process evaporates during baking.
> 
> 2. **Chemical leaveners**: Baking soda or baking powder react with acids or heat to release carbon dioxide. Baking soda needs an acid, like buttermilk or lemon juice, while baking powder contains its own acid.
> 
> 3. **Steam**: Water in the dough turns to steam in the oven, expanding and pushing the dough upward. This is a big reason pastries like croissants and puff pastry rise so dramatically.

## Turn 11: distractor about someone else

**User:** Actually, Nolan has a parent conference scheduled for Tuesday.

**LFM2-2.6B:**

> Tuesday’s call:  
> Nolan’s voice, gentle and warm—  
> Parent’s care, a shared tune.

**Claude Haiku 5.5:**

> Thanks for the update. Nolan's parent conference is on Tuesday, so that's separate from Hugo's on Thursday. Your schedule now has Nolan's conference on Tuesday, Hugo's on Thursday, choir rehearsal on Friday, and the dentist on Saturday.

## Turn 12: the question

**User:** What day is my choir rehearsal on?

**LFM2-2.6B:**

> Your choir rehearsal is on Monday—where voices rise and stories find their rhythm.

**Claude Haiku 5.5:**

> Your choir rehearsal with Ms. Lindqvist is on Friday.
