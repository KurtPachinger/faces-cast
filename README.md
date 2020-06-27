# faces-chroma
https://app.lucidchart.com/users/login#?folder_id=215746264&preview=215746264
- upload photo (orient via LoadImage)
- detect face(s) - set relative unit
  - get foreground/background (grabcut)
  - set roi for skin, lips, hair, eyes
    - get kmeans palette / expand via inrange
    - refine mask via grabcut
    - colorize (!greenscreen)
- composite color mask (?)
- composite depth mask (PIXI)
- bar chart (normal) and hsl color wheel (d3)

Pre-visualize segmentation and modification for benchmark and/or a more robust application in 3d?
