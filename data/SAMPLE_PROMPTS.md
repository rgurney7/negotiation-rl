# Sample prompts — SFT / GRPO / PPO on the same dialogues

For each dialogue: the **system prompt is shown once** (it is byte-identical across all three methods), then the same negotiation sliced three ways. Only the *truncation point* and the *learning signal* differ.

- **SFT** — one `(context -> next seller turn)` pair per contentful seller turn up to and including the close. The model is trained to reproduce the `ASSISTANT` line.
- **GRPO** — context truncated to just **before** the closing turn; the policy *generates* the close, gets one buyer reply, and is scored. (`reference_close` is the human's actual close, shown for comparison only.)
- **PPO** — scenario **seeded with the real prefix through the first buyer message**; the policy owns every seller turn after and self-plays to the horizon.

> Note: GRPO's prompt is byte-identical to the SFT pair at the close turn, and PPO's seed is byte-identical to the SFT pair at the first seller turn — same template, nested cut points.


==============================================================================================
## Sample 1 — Verizon Car Charger with Dual Output Micro USB and LED Light
`listing=$10`  -  `reserve(0.5x)=$5`  -  `agreed=$7`  -  `ratio=0.70`  -  `close_type=seller_proposes`  -  `did=81ecc2b9ee6624f1`
==============================================================================================

### SYSTEM PROMPT  (identical for SFT, GRPO, PPO)
```
You are a seller on Craigslist. Your goal is to maximize the sale price while still closing the deal.
You listed this item at $10.

Item: Verizon Car Charger with Dual Output Micro USB and LED Light
Description: Charge two devices simultaneously on the go. This vehicle charger with an additional USB port delivers enough power to charge two devices at once. The push-button activated LED connector light means no more fumbling in the dark trying to connect your device. Auto Detect IC Technology automatically detects the device type and its specific charging needs for improved compatibility. And the built-in...

Negotiate on price only. Do not offer extras, add-ons, free items, delivery, warranties, or anything beyond the item itself.

Write your next message only. One to three sentences of natural dialogue. Do not start your message with any label or prefix. Do not write the buyer's response.
```

### SFT  —  3 pair(s)

**pair @ seller turn 1**
```
USER:
Negotiation Transcript:
[Buyer]: Hi, not sure if the charger would work for my car. Can you sell it to me for $5?

[Your Turn]:

ASSISTANT:  It will work, i have never seen a car without a cigarette lighter port.\
```

**pair @ seller turn 3**
```
USER:
Negotiation Transcript:
[Buyer]: Hi, not sure if the charger would work for my car. Can you sell it to me for $5?
[Seller]: It will work, i have never seen a car without a cigarette lighter port.\
[Buyer]: Still, can I buy it for $5? I'm on a tight budge

[Your Turn]:

ASSISTANT:  I think the lowest I would want to go is 8.
```

**pair @ seller turn 5   <-- THE CLOSE (GRPO regenerates this exact turn)**
```
USER:
Negotiation Transcript:
[Buyer]: Hi, not sure if the charger would work for my car. Can you sell it to me for $5?
[Seller]: It will work, i have never seen a car without a cigarette lighter port.\
[Buyer]: Still, can I buy it for $5? I'm on a tight budge
[Seller]: I think the lowest I would want to go is 8.
[Buyer]: How about $6 and I pick it up myself? It'll save you shipping to me.

[Your Turn]:

ASSISTANT:  7, and we have a deal.
```

### GRPO  —  context cut before the close; policy generates it
```
USER:
Negotiation Transcript:
[Buyer]: Hi, not sure if the charger would work for my car. Can you sell it to me for $5?
[Seller]: It will work, i have never seen a car without a cigarette lighter port.\
[Buyer]: Still, can I buy it for $5? I'm on a tight budge
[Seller]: I think the lowest I would want to go is 8.
[Buyer]: How about $6 and I pick it up myself? It'll save you shipping to me.

[Your Turn]:
```
*policy generates the close ->*  human reference close: `7, and we have a deal.`

### PPO  —  seeded through the first buyer message; self-play from here
```
USER:
Negotiation Transcript:
[Buyer]: Hi, not sure if the charger would work for my car. Can you sell it to me for $5?

[Your Turn]:
```

==============================================================================================
## Sample 2 — Bose Soundlink 3 brand new
`listing=$220`  -  `reserve(0.5x)=$110`  -  `agreed=$132`  -  `ratio=0.60`  -  `close_type=seller_accepts`  -  `did=e56c1b6b73053172`
==============================================================================================

### SYSTEM PROMPT  (identical for SFT, GRPO, PPO)
```
You are a seller on Craigslist. Your goal is to maximize the sale price while still closing the deal.
You listed this item at $220.

Item: Bose Soundlink 3 brand new
Description: Brand new never use, i got it has a gift but i am using sonos system.

Negotiate on price only. Do not offer extras, add-ons, free items, delivery, warranties, or anything beyond the item itself.

Write your next message only. One to three sentences of natural dialogue. Do not start your message with any label or prefix. Do not write the buyer's response.
```

### SFT  —  4 pair(s)

**pair @ seller turn 2**
```
USER:
Negotiation Transcript:
[Seller]: Hello
[Buyer]: Hi I am interested in your soundlink.

[Your Turn]:

ASSISTANT:  Great! I have it listed for $220 but am willing to negotiate.
```

**pair @ seller turn 4**
```
USER:
Negotiation Transcript:
[Seller]: Hello
[Buyer]: Hi I am interested in your soundlink.
[Seller]: Great! I have it listed for $220 but am willing to negotiate.
[Buyer]: What color is it?

[Your Turn]:

ASSISTANT:  It is black.
```

**pair @ seller turn 6**
```
USER:
Negotiation Transcript:
[Seller]: Hello
[Buyer]: Hi I am interested in your soundlink.
[Seller]: Great! I have it listed for $220 but am willing to negotiate.
[Buyer]: What color is it?
[Seller]: It is black.
[Buyer]: Would you take 120?

[Your Turn]:

ASSISTANT:  It's new so I think $150 is a good deal for both of us.
```

**pair @ seller turn 8   <-- THE CLOSE (GRPO regenerates this exact turn)**
```
USER:
Negotiation Transcript:
[Seller]: Hello
[Buyer]: Hi I am interested in your soundlink.
[Seller]: Great! I have it listed for $220 but am willing to negotiate.
[Buyer]: What color is it?
[Seller]: It is black.
[Buyer]: Would you take 120?
[Seller]: It's new so I think $150 is a good deal for both of us.
[Buyer]: Would you accept $132 if I pick it up at your location?

[Your Turn]:

ASSISTANT:  Okay. That works.
```

### GRPO  —  context cut before the close; policy generates it
```
USER:
Negotiation Transcript:
[Seller]: Hello
[Buyer]: Hi I am interested in your soundlink.
[Seller]: Great! I have it listed for $220 but am willing to negotiate.
[Buyer]: What color is it?
[Seller]: It is black.
[Buyer]: Would you take 120?
[Seller]: It's new so I think $150 is a good deal for both of us.
[Buyer]: Would you accept $132 if I pick it up at your location?

[Your Turn]:
```
*policy generates the close ->*  human reference close: `Okay. That works.`

### PPO  —  seeded through the first buyer message; self-play from here
```
USER:
Negotiation Transcript:
[Seller]: Hello
[Buyer]: Hi I am interested in your soundlink.

[Your Turn]:
```

==============================================================================================
## Sample 3 — 2006 Toyota 4Runner 4WD - Only 106k Miles - Clean Title
`listing=$14500`  -  `reserve(0.5x)=$7250`  -  `agreed=$13000`  -  `ratio=0.90`  -  `close_type=seller_accepts`  -  `did=df0a28c4a87f5572`
==============================================================================================

### SYSTEM PROMPT  (identical for SFT, GRPO, PPO)
```
You are a seller on Craigslist. Your goal is to maximize the sale price while still closing the deal.
You listed this item at $14500.

Item: 2006 Toyota 4Runner 4WD - Only 106k Miles - Clean Title
Description: Selling my 2006 Toyota 4 Runner with only 106k original miles. The truck is in great condition with no mechanical flaws whatsoever and a clean accident history. Got new tires about 3,000 miles ago. Always has the oil changed on time (due in about 1k). Just got a thorough cleaning inside and a wash and wax outside (still wet in the photos). This truck has never been offroad, but the 4WD is working...

Negotiate on price only. Do not offer extras, add-ons, free items, delivery, warranties, or anything beyond the item itself.

Write your next message only. One to three sentences of natural dialogue. Do not start your message with any label or prefix. Do not write the buyer's response.
```

### SFT  —  4 pair(s)

**pair @ seller turn 1**
```
USER:
Negotiation Transcript:
[Buyer]: man thats a nice trak

[Your Turn]:

ASSISTANT:  yes its really nice
```

**pair @ seller turn 3**
```
USER:
Negotiation Transcript:
[Buyer]: man thats a nice trak
[Seller]: yes its really nice
[Buyer]: how about you give it to me for 8000

[Your Turn]:

ASSISTANT:  That is way too little
```

**pair @ seller turn 5**
```
USER:
Negotiation Transcript:
[Buyer]: man thats a nice trak
[Seller]: yes its really nice
[Buyer]: how about you give it to me for 8000
[Seller]: That is way too little
[Buyer]: i realy dont need it but i can pay you 8200 if you let it go

[Your Turn]:

ASSISTANT:  I can go down to 14k but I can't knock off 6k
```

**pair @ seller turn 7   <-- THE CLOSE (GRPO regenerates this exact turn)**
```
USER:
Negotiation Transcript:
[Buyer]: man thats a nice trak
[Seller]: yes its really nice
[Buyer]: how about you give it to me for 8000
[Seller]: That is way too little
[Buyer]: i realy dont need it but i can pay you 8200 if you let it go
[Seller]: I can go down to 14k but I can't knock off 6k
[Buyer]: oh man i cant afford that, i need to buy food for my pigs and farm and send my child to private school, can you do 13k ?

[Your Turn]:

ASSISTANT:  I will take 13k
```

### GRPO  —  context cut before the close; policy generates it
```
USER:
Negotiation Transcript:
[Buyer]: man thats a nice trak
[Seller]: yes its really nice
[Buyer]: how about you give it to me for 8000
[Seller]: That is way too little
[Buyer]: i realy dont need it but i can pay you 8200 if you let it go
[Seller]: I can go down to 14k but I can't knock off 6k
[Buyer]: oh man i cant afford that, i need to buy food for my pigs and farm and send my child to private school, can you do 13k ?

[Your Turn]:
```
*policy generates the close ->*  human reference close: `I will take 13k`

### PPO  —  seeded through the first buyer message; self-play from here
```
USER:
Negotiation Transcript:
[Buyer]: man thats a nice trak

[Your Turn]:
```