## Setup Data

Follow XTREME:

`./download_data.sh`

Need to download panx dataset manually first as `$REPO/download/AmazonPhotos.zip`.
The modified preprocessing will conserve the test labels for fast evaluation, and also save the label list as `$REPO/download/panx/labels.txt` using English training set.

XTREME first uses `panx_tokenize` task to split long example into separate examples. This process is moved into tensorization.

## Notes

* `tag_un` decides whether to compute uncertainty in model. `use_un_probs` decides whether to use the computed uncertainty in the final probability; it can be turned off at runtime even `tag_un` is true; it has no effect when `tag_un` is false. Therefore, all evaluation-related methods should take `use_un_probs` as parameter.
* For evaluation metrics only, `use_un_probs` can be turned off since it won't affect final prediction (argmax). It only matters for probs (uncertainty estimation, entropy selection, etc.).