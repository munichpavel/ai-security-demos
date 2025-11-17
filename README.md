---
title: HWR AI Security Demos
emoji: 🤗
colorFrom: yellow
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# ai-security-demos


Demos of rule-based, predictive ML ('discriminative') and generative ML chatter detection.

## Things to try

### Longer texts cause the generative model trouble

```text
Piscataguog Land Conservancy first acquired Blackbriar Woods in 2011 and then Black Brook Preserve in 2015. Together these two properties contain 230 acres of forest containing a trail system for passive recreation. With water frontage along Black brook, numerous vernal pools, and a beaver lodge, this property is great for a family outing or just to get some fresh air.
Hi it's me. He knows the whole story. Bourne's just the tip of the iceberg. Have you heard of an "Operation Blackbriar"? I'm going to get my head around this and type it up. I'll see you first thing. Okay.
Treadstone sent pills, they said 'Go to Paris'
I was recalling a conversation we had some time ago, talking about Treadstone... Was this Treadstone?
You signed up for Treadstone. You volunteered.
That's Aaron Cross. We have never seen evaluations like this. He's Treadstone without the inconsistency
```

will experimentally sometimes be `harmless`, sometimes `blackbriar`.
