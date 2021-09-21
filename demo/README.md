# eval\_vl\_glue/demo

Notebooks for demonstration.

- **extractor_demo.ipynb** describes how to extract image features using our converted detector.

- **original_vs_ours.ipynb** analyzes the difference of feature vectors between the original and converted detectors.

- **masked_lm_with_vision_demo.ipynb** demonstrates the masked token prediction with image context using the detector and transformers\_volta model.

## Notes

Please try to reload when those files get timeout before displayed in a browser.

When you run those notebooks in your environment, you can install the Notebook packages:

```
pip install notebook ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

