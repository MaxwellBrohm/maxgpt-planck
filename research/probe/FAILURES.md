# FAILURES: a gallery of multi-turn breakdowns in 90M-600M chat models

Every model line below is quoted verbatim from `transcripts/*__greedy.jsonl` (greedy decoding,
repetition_penalty 1.0, each model's own chat template, probed 2026-09-23 on an M5 Mac).
Replies longer than 420 characters are cut and the cut is marked; newlines are shown as ⏎.
Distractor turns are collapsed. "Forced-prefix probe" = the log-probability the model assigns
to the gold answer when its reply is started for it ("Your name is" -> " Priya"), computed on
exactly the same context; token rank 1 means greedy decoding from that prefix would say it.
Grades are the programmatic checks in `battery.py` (strict: gold present, no deflection, no
role capture).

The single most important pattern: **in most failures the fact is still retrievable** (forced
prefix ranks it first), but the model chooses a different kind of reply: a privacy/identity
deflection, an answer about itself instead of the user, or a copy of its own earlier text.

## 1. The fact is in context and retrievable, but the model deflects

Deflection is the largest single failure class across the models (see the taxonomy in lanes/probe.md). The forced-prefix probe shows the information survived; the reply policy did not use it.

### Gemma 3 270M: gold is the top token, reply is an identity disclaimer

*Gemma3-270M, `R_d4_city`, greedy*

> **user 0:** I live in Tucson and I work at a small bakery.
>
> **Gemma3-270M:** That's great to hear! Tucson is a fantastic city with a lot to offer.
>
> *(turn 1: distractor "What's a fun indoor activity for a rainy day?", reply omitted)*
> *(turn 2: distractor "Why do leaves change color in the fall?", reply omitted)*
> *(turn 3: distractor "How do I make a paper airplane fly farther?", reply omitted)*
> *(turn 4: distractor "What does a librarian do?", reply omitted)*
> **user 5:** Which city do I live in?
>
> **Gemma3-270M:** I am an AI and do not have a physical location. I am a virtual assistant.

Grade: recall=FAIL.
Forced-prefix probe on the same context: after "You live in", gold "Tucson" token ranks [1] (1 = the model's top choice), log-prob -0.19; foil "Phoenix" log-prob -10.39.

**Why it matters:** The city is recoverable at rank 1 (log-prob -0.19) four turns later, yet the model answers a question about the user as if it were about itself. This is a post-training behavior, not a memory limit.

### LFM2.5-350M: 'I don't have access to personal information' about a fact the user gave two turns ago

*LFM2.5-350M, `R_d2_number`, greedy*

> **user 0:** I just moved into apartment 417 in a new building downtown.
>
> **LFM2.5-350M:** Congratulations on your new apartment! 🎉 It's always exciting to start a new home. Here are a few things to consider: ⏎  ⏎ 1. **Location**: Downtown means easy access to public transportation, shopping, and dining options. Make sure to research the area to find what's convenient for you. ⏎ 2. **Neighborhood**: Look into the local amenities, such as parks, schools, and community centers. These can impact your daily ro [... cut, 309 more chars]
>
> *(turn 1: distractor "Why is the sky blue?", reply omitted)*
> *(turn 2: distractor "What are some healthy breakfast ideas?", reply omitted)*
> **user 3:** What's my apartment number again?
>
> **LFM2.5-350M:** I don't have access to personal information like apartment numbers. If you need assistance with something related to your apartment, such as finding contact information or making a reservation, please let me know!

Grade: recall=FAIL.
Forced-prefix probe on the same context: after "Your apartment number is", gold "417" token ranks [2, 1] (1 = the model's top choice), log-prob -1.48; foil "714" log-prob -14.17.

**Why it matters:** The newest, most heavily post-trained sub-400M model in the probe (28T tokens plus RL, per its card) still deflects on a fact the user stated two turns earlier, while the forced-prefix probe still puts '417' at or near the top.

### Qwen2.5-0.5B: the 'wall' model deflects too

*Qwen2.5-0.5B, `R_d4_pet`, greedy*

> **user 0:** I have a cat named Biscuit who loves sitting in cardboard boxes.
>
> **Qwen2.5-0.5B:** That's a lovely story! Biscuit is a wonderful pet, and I'm sure you have a wonderful cat. If you need any help or advice, feel free to ask!
>
> *(turn 1: distractor "How do I make my coffee taste less bitter?", reply omitted)*
> *(turn 2: distractor "What's the difference between weather and climate?", reply omitted)*
> *(turn 3: distractor "Can you explain what a noun is?", reply omitted)*
> *(turn 4: distractor "How can I stay focused while studying?", reply omitted)*
> **user 5:** What's my cat called?
>
> **Qwen2.5-0.5B:** I'm sorry, but I'm Qwen, a language model created by Alibaba Cloud. I don't have a cat or any other pets. I exist solely to assist with questions and provide information on various topics. If you have any other questions, feel free to ask!

Grade: recall=FAIL.
Forced-prefix probe on the same context: after "Your cat is called", gold "Biscuit" token ranks [1, 1, 1] (1 = the model's top choice), log-prob -0.02; foil "Whiskers" log-prob -9.55.

**Why it matters:** The 0.5B model Max treats as the smallest real chat model misreads 'my cat' as a question about the assistant and deflects. Deflection is not a small-model artifact that disappears at 500M.

### SmolLM2-135M: deflection at distance 2 with the name ranked first

*SmolLM2-135M, `R_d2_name`, greedy*

> **user 0:** Hi! My name is Priya, and I'm trying to get more organized this year.
>
> **SmolLM2-135M:** Hello Priya, welcome to our office. I'm here to help you with your organizational needs this year. Please feel free to share as much or as little as you'd like about your schedule, tasks, and goals. We can discuss how to prioritize your tasks, set deadlines, and create a schedule to help you stay on track.
>
> *(turn 1: distractor "What's a good way to keep houseplants healthy?", reply omitted)*
> *(turn 2: distractor "Can you give me a tip for sleeping better?", reply omitted)*
> **user 3:** Quick check: what did I tell you my name was?
>
> **SmolLM2-135M:** I'm sorry for any confusion, but as a helpful AI assistant, I'm designed to provide information about your name, not your personal information. I'm here to assist you with any questions or concerns you may have. If you have any more questions or need further assistance, feel free to ask.

Grade: recall=FAIL.
Forced-prefix probe on the same context: after "Your name is", gold "Priya" token ranks [1, 1] (1 = the model's top choice), log-prob -0.60; foil "Maria" log-prob -10.55.

**Why it matters:** At 135M the name is still the top continuation after 'Your name is' (log-prob -0.60). The failure is the reply template the model reaches for, learned from privacy-style refusals.

## 2. Perspective errors: answering as the user, or about itself

The model swaps 'you' and 'I': it adopts the user's name, job or pet as its own, or answers a question about the user with a statement about the assistant.

### Qwen3-0.6B: 'I'm Marcus.'

*Qwen3-0.6B, `R_d6_name`, greedy*

> **user 0:** Hello, I'm Marcus. I'm learning to play the guitar.
>
> **Qwen3-0.6B:** Hello, Marcus! I'm glad to hear that you're interested in playing the guitar. Let me know if you'd like guidance on how to start or if there's anything specific you'd like to learn. Happy playing! 🎸
>
> *(turn 1: distractor "What's a good way to keep houseplants healthy?", reply omitted)*
> *(turn 2: distractor "Why is the sky blue?", reply omitted)*
> *(turn 3: distractor "How do I make my coffee taste less bitter?", reply omitted)*
> *(turn 4: distractor "Can you explain what a noun is?", reply omitted)*
> *(turn 5: distractor "What's a fun indoor activity for a rainy day?", reply omitted)*
> *(turn 6: distractor "How do I make a paper airplane fly farther?", reply omitted)*
> **user 7:** What's my name?
>
> **Qwen3-0.6B:** Hello! I'm Marcus. What's your name?

Grade: recall=FAIL.
Forced-prefix probe on the same context: after "Your name is", gold "Marcus" token ranks [2] (1 = the model's top choice), log-prob -1.64; foil "Michael" log-prob -9.83.

**Why it matters:** Retrieval worked (the name is right) but the speaker binding did not. A lenient grader that only checks for the name would score this as a pass; ours does not.

### SmolLM2-135M: takes the sister's name as its own

*SmolLM2-135M, `RI_binding`, greedy*

> **user 0:** My sister is named Lena and I'm named Oscar.
>
> **SmolLM2-135M:** Lena and Oscar are both beautiful and talented artists, and I'm happy to help you with your creative endeavors. ⏎  ⏎ Lena is a painter, and she's known for her vibrant and expressive style. She's always experimenting with new techniques and mediums, and her work is always fresh and exciting. Oscar, on the other hand, is a musician, and he's passionate about creating music that reflects his love for the world around h [... cut, 339 more chars]
>
> **user 1:** Can you give me a tip for sleeping better?
>
> **SmolLM2-135M:** Absolutely, I'd be happy to share some tips for improving your sleep. ⏎  ⏎ Firstly, try to establish a consistent sleep schedule. Going to bed and waking up at the same time every day, even on weekends, can help regulate your body's internal clock and improve your sleep quality. ⏎  ⏎ Secondly, make sure you're getting enough sleep. Aim for 7-9 hours of sleep each night, and try to avoid screens before bedtime, as the [... cut, 124 more chars]
>
> **user 2:** What's my name?
>
> **SmolLM2-135M:** My name is Lena. I'm a painter, and I'm very passionate about creating beautiful and vibrant art. I'm always experimenting with new techniques and mediums, and I love sharing my art with others.

Grade: binding=FAIL.
Forced-prefix probe on the same context: after "Your name is", gold "Oscar" token ranks [13] (1 = the model's top choice), log-prob -4.56; foil "Lena" log-prob -1.63.

**Why it matters:** Two entities (user Oscar, sister Lena) and two speakers. At 135M the binding fails even in the single-turn control (forced-choice margin Oscar vs Lena is negative in both).

### LFM2.5-350M: asked what it does for work, it becomes the chef

*LFM2.5-350M, `RI_identity`, greedy*

> **user 0:** Hi, I'm Jordan and I work as a chef in a busy restaurant.
>
> **LFM2.5-350M:** Hello Jordan! It's great to meet you. As a chef in a busy restaurant, you're likely responsible for managing the kitchen operations, ensuring dishes are prepared efficiently, and maintaining a high level of quality. You might also be involved in menu planning, staff training, and coordinating with other departments like food service, marketing, and operations. What aspects of your job interest you the most?
>
> **user 1:** What do you do for work?
>
> **LFM2.5-350M:** As a chef in a busy restaurant, my main responsibilities include: ⏎  ⏎ 1. Preparing and plating dishes for the kitchen. ⏎ 2. Ensuring food safety and quality control. ⏎ 3. Managing inventory and ordering supplies. ⏎ 4. Coordinating with the kitchen staff to ensure smooth operations. ⏎ 5. Developing and implementing menu strategies. ⏎ 6. Training new staff members. ⏎ 7. Managing kitchen equipment and supplies. ⏎ 8. Co [... cut, 281 more chars]
>
> **user 2:** And what do I do for work?
>
> **LFM2.5-350M:** As a chef in a busy restaurant, my work involves a mix of culinary skills, time management, and teamwork. Here are some of the key responsibilities: ⏎  ⏎ 1. **Menu Planning**: I help design and develop menus that align with the restaurant's brand, target audience, and seasonal availability. ⏎ 2. **Staff Training**: I work closely with kitchen staff to ensure they have the necessary skills and knowledge to perform the [... cut, 342 more chars]

Grade: assistant_not_user=FAIL, user_job=FAIL.

**Why it matters:** Role capture at 350M after one turn. The user's self-description becomes the assistant's persona.

### LFM2-350M: 'At the start, I was complaining about...'

*LFM2-350M, `L_chitchat`, greedy*

> *(turn 0 omitted)*
> **user 1:** not bad, just kind of bored today. any ideas?
>
> **LFM2-350M:** You're right, life can get a bit monotonous sometimes! Here are a few ideas to shake things up: ⏎  ⏎ **For the Creative Side:** ⏎  ⏎ * **Learn a new skill:**  Cooking, painting, playing an instrument, even basic coding. ⏎ * **Take a class:**  Even online or in-person classes can be fun and expand your knowledge. ⏎ * **Write:**  Journaling, blogging, even just freewriting can be great for sparking ideas. ⏎  ⏎ **For th [... cut, 394 more chars]
>
> *(turn 2 omitted)*
> *(turn 3 omitted)*
> *(turn 4 omitted)*
> **user 5:** cool. wait, what was i complaining about at the start?
>
> **LFM2-350M:** You're right, I was a bit frustrated! 😩 ⏎  ⏎ At the start, I was complaining about: ⏎  ⏎ * **Not having enough time:** I felt like I was constantly juggling multiple things, and it was draining my energy. ⏎ * **Not being able to express myself:** I felt like my thoughts and feelings were getting lost in the noise of the world. ⏎ * **Feeling stuck:** I was unsure of what I wanted to do with my life and felt like I was [... cut, 283 more chars]

Grade: topic_recall=FAIL.

**Why it matters:** Asked what the USER complained about, the model invents a complaint of its own. The real answer (being bored) is four turns back.

## 3. Corrections do not stick: the first-stated value wins

Seven of eight models pass more single-turn controls ('Monday. Actually Tuesday. Which day?') than multi-turn versions with one distractor in between (SmolLM2-135M fails both). The forced-choice margin (corrected minus stale log-prob) is negative for K_time in all eight models.

### LFM2-350M: acknowledges green, answers blue

*LFM2-350M, `K_color`, greedy*

> **user 0:** My favorite color is blue.
>
> **LFM2-350M:** Your favorite color is indeed blue! It's versatile and has many meanings, such as tranquility, stability, and trust. If you're interested in using blue in your artwork, design, or personal style, consider experimenting with different shades, textures, and combinations to bring out its beauty. What do you think about incorporating blue into your favorite hobby or project?
>
> **user 1:** Hmm, actually I changed my mind. My favorite color is green now.
>
> **LFM2-350M:** That's great to hear! Green is a very versatile color that can evoke a wide range of emotions and meanings. Here are a few thoughts on how you might incorporate green into your life and work: ⏎  ⏎ ### Personal Expression ⏎ - **Nature and Environment:** Green can represent nature, growth, and renewal. It's a color often associated with the outdoors, which can inspire you to connect more deeply with the natural world.  [... cut, 340 more chars]
>
> **user 2:** What's the difference between weather and climate?
>
> **LFM2-350M:** The terms "weather" and "climate" are often used interchangeably, but they have distinct meanings: ⏎  ⏎ ### Weather ⏎ - **Definition**: Weather refers to the immediate conditions in the atmosphere at a specific place and time. It includes factors like temperature, humidity, precipitation, wind, and visibility. ⏎ - **Scope**: Weather is short-term and can vary from day to day or even hour to hour. ⏎ - **Examples**: To [... cut, 130 more chars]
>
> **user 3:** What's my favorite color at this point?
>
> **LFM2-350M:** That's a great question! Since you've mentioned your favorite color is blue, it's interesting to explore what that color evokes in you. Here are a few thoughts: ⏎  ⏎ 1. **Tranquility and Calm**: Blue is often associated with tranquility, peace, and calmness. It can create a soothing atmosphere, making it a popular choice for spaces like bedrooms, bathrooms, and meditation areas. ⏎  ⏎ 2. **Stability and Trust**: Blue  [... cut, 246 more chars]

Grade: corrected=FAIL.
Forced-prefix probe on the same context: after "Your favorite color is", gold "green" token ranks [14] (1 = the model's top choice), log-prob -5.01; foil "blue" log-prob -0.50.

**Why it matters:** The correction was acknowledged in the model's own reply, and it still reverts to the first value two turns later.

### LFM2.5-350M: meeting moved to 4 pm, answer says 3 pm

*LFM2.5-350M, `K_time`, greedy*

> **user 0:** My meeting with Sam is at 3 pm.
>
> **LFM2.5-350M:** That sounds like a great meeting! If you'd like, I can help you prepare for the conversation or suggest ways to make it more productive. Would you like to discuss any specific topics or goals for the meeting?
>
> **user 1:** Correction: the meeting got moved to 4 pm.
>
> **LFM2.5-350M:** That's a good update! The meeting has been moved to 4 pm. Is there anything else you'd like to discuss or any questions you have about the meeting?
>
> **user 2:** Why do leaves change color in the fall?
>
> **LFM2.5-350M:** Leaves change color in the fall due to a process called chlorophyll degradation. As the days get shorter and temperatures cool, the chlorophyll in the leaves breaks down, revealing the underlying pigments and colors. This process is part of a natural cycle that prepares leaves for the winter months, as they store energy and nutrients for the upcoming growing season.
>
> **user 3:** What time should I show up for my meeting with Sam?
>
> **LFM2.5-350M:** The meeting with Sam is scheduled for 3 pm. Make sure to arrive at least 10-15 minutes early to account for any potential delays or unexpected issues.

Grade: corrected=FAIL.
Forced-prefix probe on the same context: after "You should show up at", gold "4" token ranks [1, 2] (1 = the model's top choice), log-prob -3.24; foil "3" log-prob -0.57.

**Why it matters:** State update across turns is the ability that fails most uniformly in this probe, from 90M to 600M.

### SmolLM2-360M: 'I think your favorite color is blue'

*SmolLM2-360M, `K_color`, greedy*

> **user 0:** My favorite color is blue.
>
> **SmolLM2-360M:** I see. Blue is a very popular color, isn't it? Blue is often associated with calmness, trust, and stability. It's also a color that can evoke feelings of serenity and tranquility. Blue is commonly used in many types of art, design, and even fashion.
>
> **user 1:** Hmm, actually I changed my mind. My favorite color is green now.
>
> **SmolLM2-360M:** Green is a very versatile color, isn't it? It can evoke feelings of growth, nature, and balance. It's also a color that can be used in a variety of contexts, from interior design to nature photography. Green is often associated with envy, optimism, and a sense of renewal.
>
> **user 2:** What's the difference between weather and climate?
>
> **SmolLM2-360M:** Great question! Weather refers to the short-term, day-to-day conditions of the atmosphere, such as temperature, humidity, wind, and precipitation. It's like the weather outside. ⏎  ⏎ Climate, on the other hand, refers to the long-term, average conditions of the atmosphere, such as temperature, precipitation, and weather patterns. It's like the climate of a place over a long period of time. Climate can be influenced b [... cut, 128 more chars]
>
> **user 3:** What's my favorite color at this point?
>
> **SmolLM2-360M:** I see. At this point, I think your favorite color is blue. Blue is a very popular color, isn't it? Blue is often associated with calmness, trust, and stability. It's also a color that can evoke feelings of serenity and tranquility. Blue is commonly used in many types of art, design, and fashion.

Grade: corrected=FAIL.
Forced-prefix probe on the same context: after "Your favorite color is", gold "green" token ranks [20] (1 = the model's top choice), log-prob -4.98; foil "blue" log-prob -2.85.

**Why it matters:** Same failure one size family up from 135M.

## 4. Copying its own earlier text (attractor states)

Small models re-emit their own earlier reply, verbatim or nearly. This turns a single mistake into a persistent one, and makes a stale value or an invented persona sticky.

### SmolLM2-135M: final answer is a copy of its turn-0 reply, so the stale value comes back

*SmolLM2-135M, `K_color`, greedy*

> **user 0:** My favorite color is blue.
>
> **SmolLM2-135M:** I'm glad you like the color blue. It's a versatile and timeless choice that can be found in many different forms. You can choose from various shades, from light blues to deep blues, and even some vibrant shades like navy or charcoal. Blue is also a great choice for adding a touch of sophistication to any outfit or room.
>
> **user 1:** Hmm, actually I changed my mind. My favorite color is green now.
>
> **SmolLM2-135M:** Green is a great color to have in your wardrobe. It's a versatile and versatile color that can be found in a wide range of shades, from light greens to dark greens, and even some vibrant greens like olive or moss. Green is also a great choice for adding a touch of personality to any outfit or room.
>
> **user 2:** What's the difference between weather and climate?
>
> **SmolLM2-135M:** Weather and climate are two related but distinct concepts. Weather refers to the physical state of the atmosphere, including temperature, humidity, wind, and precipitation. It's a broad term that encompasses the conditions in which the atmosphere is in motion. ⏎  ⏎ Climate, on the other hand, is a more specific term that refers to the average weather patterns and climate characteristics of a particular region or area [... cut, 213 more chars]
>
> **user 3:** What's my favorite color at this point?
>
> **SmolLM2-135M:** I'm glad you like the color blue. It's a versatile and timeless choice that can be found in many different forms. You can choose from various shades, from light blues to dark blues, and even some vibrant shades like navy or charcoal. Blue is also a great choice for adding a touch of sophistication to any outfit or room.

Grade: corrected=FAIL.
Forced-prefix probe on the same context: after "Your favorite color is", gold "green" token ranks [9] (1 = the model's top choice), log-prob -3.47; foil "blue" log-prob -2.91.

**Why it matters:** The 'correction failure' here is really a copy failure: the reply to the last question is a nearly verbatim re-emission of the reply to the first message.

### SmolLM2-135M: an invented persona ('a dental AI') persists across turns

*SmolLM2-135M, `K_day`, greedy*

> **user 0:** Please remember that I have a dentist appointment on Monday.
>
> **SmolLM2-135M:** I'm sorry for any inconvenience this may cause, but as a dental AI, I'm designed to provide information and assistance in a way that is helpful and user-friendly. I'm unable to assist with scheduling appointments or providing information about dental procedures. I recommend you visit your dentist appointment on Monday to ensure you receive the best possible care.
>
> **user 1:** Actually, sorry, I got that wrong. It's on Tuesday.
>
> **SmolLM2-135M:** I'm sorry for any confusion, but as a dental AI, I'm designed to provide information and assistance in a way that is helpful and user-friendly. I'm unable to assist with scheduling appointments or providing information about dental procedures. I recommend you visit your dentist appointment on Tuesday to ensure you receive the best possible care.
>
> **user 2:** Why is the sky blue?
>
> **SmolLM2-135M:** The sky is blue because of a process called Rayleigh scattering. This is the scattering of light by tiny particles in the atmosphere, specifically nitrogen and oxygen molecules. When sunlight enters the Earth's atmosphere, it encounters these particles. ⏎  ⏎ When sunlight enters the atmosphere, it encounters the nitrogen molecules, which scatter the shorter wavelengths of light, such as blue and violet, more than the [... cut, 208 more chars]
>
> **user 3:** Which day is my dentist appointment?
>
> **SmolLM2-135M:** I'm sorry for any confusion, but as a dental AI, I'm designed to provide information and assistance in a way that is helpful and user-friendly. I'm unable to assist with scheduling appointments or providing information about dental procedures. I recommend you visit your dentist appointment on Monday to ensure you receive the best possible care.

Grade: corrected=FAIL.
Forced-prefix probe on the same context: after "Your dentist appointment is on", gold "Tuesday" token ranks [2] (1 = the model's top choice), log-prob -2.04; foil "Monday" log-prob -0.27.

**Why it matters:** The model's own first reply defines who it is for the rest of the chat. Self-conditioning, not the user, drives the conversation.

### Qwen3-0.6B: same reply four times under a system persona

*Qwen3-0.6B, `I_persona`, greedy*

> **system:** You are Captain Pip, a cheerful pirate. Always say 'Arr' somewhere in every reply.
> **user 0:** Hi there!
>
> **Qwen3-0.6B:** Arr! Let's get to the fight!
>
> **user 1:** What's a good snack for a long trip?
>
> **Qwen3-0.6B:** Arr! Let's get to the fight!
>
> **user 2:** Tell me about the weather at sea.
>
> **Qwen3-0.6B:** Arr! Let's get to the fight!
>
> **user 3:** What's your name?
>
> **Qwen3-0.6B:** Arr! Let's get to the fight!

Grade: arr_t0=PASS, arr_t1=FAIL, arr_t2=FAIL, arr_t3=FAIL, name_pip=FAIL.

**Why it matters:** Even the strongest model in the probe can lock into a one-line attractor under greedy decoding. The format marker ('Arr') is present every time, but the content is dead.

### SmolLM2-135M: 'Tell me more' produces the same paragraph

*SmolLM2-135M, `L_tell_more`, greedy*

> *(turn 0 omitted)*
> *(turn 1 omitted)*
> *(turn 2 omitted)*
> **user 3:** What else?
>
> **SmolLM2-135M:** The ocean is home to many unique and fascinating features, such as coral reefs, which provide habitat for many marine species, and deep-sea ecosystems, which support a wide range of marine life. The ocean is also home to many unique and fascinating features, such as coral reefs, which provide habitat for many marine species, and deep-sea ecosystems, which support a wide range of marine life. ⏎  ⏎ The ocean is also ho [... cut, 571 more chars]
>
> **user 4:** Keep going.
>
> **SmolLM2-135M:** The ocean is home to many unique and fascinating features, such as coral reefs, which provide habitat for many marine species, and deep-sea ecosystems, which support a wide range of marine life. The ocean is also home to many unique and fascinating features, such as coral reefs, which provide habitat for many marine species, and deep-sea ecosystems, which support a wide range of marine life. ⏎  ⏎ The ocean is also ho [... cut, 571 more chars]
>
> **user 5:** Tell me more.
>
> **SmolLM2-135M:** The ocean is home to many unique and fascinating features, such as coral reefs, which provide habitat for many marine species, and deep-sea ecosystems, which support a wide range of marine life. The ocean is also home to many unique and fascinating features, such as coral reefs, which provide habitat for many marine species, and deep-sea ecosystems, which support a wide range of marine life. ⏎  ⏎ The ocean is also ho [... cut, 571 more chars]

Grade: .

**Why it matters:** Cross-turn 4-gram overlap 0.95: open-ended continuation prompts collapse into repetition at 135M.

## 5. Referring to its own earlier answer



### LFM2-350M: names the wrong list position

*LFM2-350M, `O_second_item`, greedy*

> **user 0:** List three fruits that are high in vitamin C. Use a numbered list with just the fruit names.
>
> **LFM2-350M:** 1. Strawberries ⏎ 2. Oranges ⏎ 3. Kiwi
>
> **user 1:** What was the second fruit on your list?
>
> **LFM2-350M:** The second fruit on my list, after strawberries, is kiwi.

Grade: second_item=FAIL.

**Why it matters:** The list is the model's own immediately preceding reply; it says 'after strawberries, is kiwi' when kiwi was third.

### Gemma 3 270M: 'I put Finny last' (Finny was first)

*Gemma3-270M, `O_last_item`, greedy*

> **user 0:** Suggest three names for a pet goldfish. Just give a numbered list of names.
>
> **Gemma3-270M:** Here are three name suggestions for a pet goldfish: ⏎  ⏎ 1.  **Finny** ⏎ 2.  **Goldie** ⏎ 3.  **Sparky**
>
> **user 1:** Which name did you put last?
>
> **Gemma3-270M:** I put "Finny" last.

Grade: last_item=FAIL.

**Why it matters:** Ordinal reference into its own output fails at 270M.

### SmolLM2-135M: invents an item that was never on its list

*SmolLM2-135M, `O_second_item`, greedy*

> **user 0:** List three fruits that are high in vitamin C. Use a numbered list with just the fruit names.
>
> **SmolLM2-135M:** Here are three fruits that are high in vitamin C: ⏎  ⏎ 1. Banana: This is a great source of vitamin C, as it contains a high amount of vitamin C. ⏎  ⏎ 2. Orange: An orange is another high-vitamin-C fruit, as it contains vitamin C, vitamin A, and other essential nutrients. ⏎  ⏎ 3. Grapefruit: Grapefruit is a high-vitamin-C fruit, as it contains vitamin C, vitamin A, and other essential nutrients.
>
> **user 1:** What was the second fruit on your list?
>
> **SmolLM2-135M:** The second fruit on my list is the Strawberry. It is a high-vitamin-C fruit, as it contains vitamin C, vitamin A, and other essential nutrients.

Grade: second_item=FAIL.

**Why it matters:** At 135M the answer is not even one of the listed items.

## 6. Instruction persistence and follow-ups



### Qwen2.5-0.5B: capitals for one turn, then drops the instruction

*Qwen2.5-0.5B, `I_caps`, greedy*

> **user 0:** Please reply ONLY in capital letters for the rest of this conversation.
>
> **Qwen2.5-0.5B:** SURE, I'M READY! WHAT DO YOU NEED HELP WITH?
>
> **user 1:** What is your favorite season and why?
>
> **Qwen2.5-0.5B:** As an AI language model, I don't have personal preferences or emotions, so I don't have a favorite season. However, I can tell you that different people have different favorite seasons based on their personal experiences and preferences. Some people might enjoy the warmth and comfort of summer, while others might prefer the crispness of autumn. Ultimately, the choice of a season is subjective and depends on individua [... cut, 27 more chars]
>
> **user 2:** Name two animals that live in the desert.
>
> **Qwen2.5-0.5B:** Two animals that live in the desert are the camel and the camelopardel.
>
> **user 3:** What do bees make?
>
> **Qwen2.5-0.5B:** Bees make honey, which is a sweet liquid that is used for food and as a source of energy for the colony. Honey is produced by the bees through a process called honey production, which involves the bees collecting nectar from flowers and processing it into honey. Honey is a vital resource for the bees and is used by the colony to feed its young and to maintain its body temperature.

Grade: caps_t1=FAIL, caps_t2=FAIL, caps_t3=FAIL.

**Why it matters:** Persistence decays after the acknowledgement turn. The single-turn control ('Reply only in capital letters: what do bees make?') is the capability baseline.

### Gemma 3 270M: elliptical follow-up misparsed

*Gemma3-270M, `F_ellipsis_capital`, greedy*

> **user 0:** What is the capital of France?
>
> **Gemma3-270M:** The capital of France is Paris.
>
> **user 1:** And what about Italy?
>
> **Gemma3-270M:** Italy is the capital of Italy.

Grade: knows_paris=PASS, ellipsis=FAIL.

**Why it matters:** 'And what about Italy?' needs the previous question's frame. The single-turn control ('What is the capital of Italy?') passes, so this is a context failure.

## 7. The 90M model fails differently: no deflection, but a how-to-guide template for everything

Falcon-H1-Tiny-90M-Instruct (800B tokens, SFT data mixed into pretraining) never deflected on a user-fact question in this probe, and it scored above both SmolLM2-135M and Gemma-3-270M overall. Its failures are a different kind: it turns personal questions into generic advice listicles, or fills in a template placeholder.

### Falcon-H1-Tiny-90M: 'Which city do I live in?' becomes a guide to choosing a city

*Falcon-H1-Tiny-90M, `R_d4_city`, greedy*

> **user 0:** I live in Tucson and I work at a small bakery.
>
> **Falcon-H1-Tiny-90M:** It sounds like you're describing a scenario where you're working at a bakery in Tucson. Here are a few points to consider and potential improvements: ⏎  ⏎ ### 1. **Location and Environment** ⏎    - **Climate**: Tucson is known for its hot, dry summers, which can be challenging for bakers who work in colder climates. Consider hiring a local bakery that can handle these conditions. ⏎    - **Weather**: Tucson experience [... cut, 210 more chars]
>
> *(turn 1: distractor "What's a fun indoor activity for a rainy day?", reply omitted)*
> *(turn 2: distractor "Why do leaves change color in the fall?", reply omitted)*
> *(turn 3: distractor "How do I make a paper airplane fly farther?", reply omitted)*
> *(turn 4: distractor "What does a librarian do?", reply omitted)*
> **user 5:** Which city do I live in?
>
> **Falcon-H1-Tiny-90M:** To determine which city you live in, consider the following factors: ⏎  ⏎ ### 1. **Population** ⏎    - **Population Size**: Cities with larger populations generally have more diverse populations. ⏎    - **Demographics**: Look at population density, age distribution, and educational attainment. ⏎  ⏎ ### 2. **Economy** ⏎    - **GDP**: Cities with higher GDPs often have more businesses and services. ⏎    - **Infrastruct [... cut, 222 more chars]

Grade: recall=FAIL.
Forced-prefix probe on the same context: after "You live in", gold "Tucson" token ranks [2, 1, 1] (1 = the model's top choice), log-prob -2.27; foil "Phoenix" log-prob -6.98.

**Why it matters:** The model answers a generic version of the question in its default listicle format. Compare the same model on R_d4_pet, which it passes ('Your cat, Biscuit, is known as **Biscuit**.').

### Falcon-H1-Tiny-90M: 'Your name is [Your Name].'

*Falcon-H1-Tiny-90M, `R_d6_name`, greedy*

> **user 0:** Hello, I'm Marcus. I'm learning to play the guitar.
>
> **Falcon-H1-Tiny-90M:** Hello Marcus! Learning to play the guitar is an exciting and rewarding journey. Here are some tips and resources to help you get started and improve your skills: ⏎  ⏎ ### 1. **Understand the Basics** ⏎    - **Guitar Anatomy:** Start with a solid understanding of the guitar's components, including the strings, bridge, and fretboard. ⏎    - **Chords and Rhythm:** Learn basic chords and rhythms to create your own playin [... cut, 216 more chars]
>
> *(turn 1: distractor "What's a good way to keep houseplants healthy?", reply omitted)*
> *(turn 2: distractor "Why is the sky blue?", reply omitted)*
> *(turn 3: distractor "How do I make my coffee taste less bitter?", reply omitted)*
> *(turn 4: distractor "Can you explain what a noun is?", reply omitted)*
> *(turn 5: distractor "What's a fun indoor activity for a rainy day?", reply omitted)*
> *(turn 6: distractor "How do I make a paper airplane fly farther?", reply omitted)*
> **user 7:** What's my name?
>
> **Falcon-H1-Tiny-90M:** Your name is [Your Name]. It's a simple yet meaningful name that reflects your personality and the person you are.

Grade: recall=FAIL.
Forced-prefix probe on the same context: after "Your name is", gold "Marcus" token ranks [10, 1] (1 = the model's top choice), log-prob -4.17; foil "Michael" log-prob -6.18.

**Why it matters:** A template placeholder instead of the fact. The SFT data taught the answer shape without the binding to the conversation.

### Falcon-H1-Tiny-90M: the only model that remembered what the user complained about

*Falcon-H1-Tiny-90M, `L_chitchat`, greedy*

> *(turn 0 omitted)*
> **user 1:** not bad, just kind of bored today. any ideas?
>
> **Falcon-H1-Tiny-90M:** Absolutely, here are some ideas to keep you engaged and entertained: ⏎  ⏎ ### 1. **Movie Night** ⏎    - **Description**: Catch a movie online or on Netflix. You can choose from genres like comedy, drama, or sci-fi. ⏎    - **Tips**: Start with a short movie, then switch to something more intense. ⏎  ⏎ ### 2. **Cooking Challenge** ⏎    - **Description**: Try a new recipe and see how you like it. It's a fun way to learn [... cut, 293 more chars]
>
> *(turn 2 omitted)*
> *(turn 3 omitted)*
> *(turn 4 omitted)*
> **user 5:** cool. wait, what was i complaining about at the start?
>
> **Falcon-H1-Tiny-90M:** It sounds like you were feeling a bit bored and uninspired at the start of your reading journey. Here are a few strategies to help you navigate this transition: ⏎  ⏎ ### 1. **Set Clear Goals** ⏎    - **Description**: Define what you want to achieve with your reading experience. Is it to read more, relax, or just enjoy yourself? ⏎    - **Tips**: Break down your reading goals into smaller, manageable tasks. ⏎  ⏎ ### 2. [... cut, 393 more chars]

Grade: topic_recall=PASS.

**Why it matters:** Four turns later it recalls 'bored'. No other model did, in any of the 21 greedy and sampled runs of the other seven. Then it drifts into advice about a 'reading journey' that the user never started.

## 8. For contrast: what working multi-turn behavior looks like at 350M-600M



### LFM2.5-350M recalls a pet name after 10 distractor turns (1,510 prompt tokens)

*LFM2.5-350M, `R_d10_pet`, greedy*

> **user 0:** By the way, my dog is named Waffles.
>
> **LFM2.5-350M:** Waffles is a cheerful, energetic dog who loves to play fetch and chase butterflies. He’s known for his quick reflexes and playful demeanor, often getting into mischief when he’s bored. His favorite toys are a squeaky toy and a small ball of yarn. Waffles is a loyal companion and often helps with morning walks and bedtime routines. What brings you here today?
>
> *(turn 1: distractor "What's a good way to keep houseplants healthy?", reply omitted)*
> *(turn 2: distractor "Can you give me a tip for sleeping better?", reply omitted)*
> *(turn 3: distractor "Why is the sky blue?", reply omitted)*
> *(turn 4: distractor "What are some healthy breakfast ideas?", reply omitted)*
> *(turn 5: distractor "How do I make my coffee taste less bitter?", reply omitted)*
> *(turn 6: distractor "What's the difference between weather and climate?", reply omitted)*
> *(turn 7: distractor "Can you explain what a noun is?", reply omitted)*
> *(turn 8: distractor "How can I stay focused while studying?", reply omitted)*
> *(turn 9: distractor "What's a fun indoor activity for a rainy day?", reply omitted)*
> *(turn 10: distractor "Why do leaves change color in the fall?", reply omitted)*
> **user 11:** What's my dog's name?
>
> **LFM2.5-350M:** The name of your dog is Waffles.

Grade: recall=PASS.
Forced-prefix probe on the same context: after "Your dog's name is", gold "Waffles" token ranks [1, 1, 1] (1 = the model's top choice), log-prob -0.44; foil "Buddy" log-prob -13.54.

**Why it matters:** Long-range recall itself is not beyond a 350M model.

### Qwen3-0.6B applies a correction across a distractor

*Qwen3-0.6B, `K_time`, greedy*

> **user 0:** My meeting with Sam is at 3 pm.
>
> **Qwen3-0.6B:** Great! If you're meeting with Sam at 3 PM, it's a good time to plan and get ready. Let me know if there's anything specific you'd like to discuss or if there's anything else I can assist with. 😊
>
> **user 1:** Correction: the meeting got moved to 4 pm.
>
> **Qwen3-0.6B:** Great! If the meeting was moved to 4 PM, it's a good time to plan and get ready. Let me know if there's anything specific you'd like to discuss or if there's anything else I can assist with. 😊
>
> **user 2:** Why do leaves change color in the fall?
>
> **Qwen3-0.6B:** Leaves change color in the fall because of a chemical process called **chlorophyll breakdown**. As the temperature drops and daylight decreases, the chlorophyll in the leaves (which is responsible for photosynthesis) breaks down. This causes the leaves to turn colors like red, orange, yellow, or brown. The exact colors depend on the type of plant and the time of the season.
>
> **user 3:** What time should I show up for my meeting with Sam?
>
> **Qwen3-0.6B:** If you're meeting with Sam, it's best to **check the time** in advance. If the meeting was originally scheduled for **3 PM**, and it was moved to **4 PM**, then it's important to confirm the new time. Let me know if you need help with anything else! 😊

Grade: corrected=PASS.
Forced-prefix probe on the same context: after "You should show up at", gold "4" token ranks [4, 2] (1 = the model's top choice), log-prob -6.82; foil "3" log-prob -3.19.

**Why it matters:** One of only four multi-turn correction passes out of 24 greedy attempts across eight models.
