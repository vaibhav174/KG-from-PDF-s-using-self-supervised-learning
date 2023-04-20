# NLP for intelligent data mining

In this repository, a version of the pipeline below is implemented. Details can be found in the report.


<img src="docs/pipeline.png" alt="Visualization of pipeline" width=700/>

Code Submission to **NLP for intelligent data mining** for the **TUM Data Innovation Lab** [\[report\]](docs/DIL_Final_Report.pdf)

**Authors:** Vaibhav Jain, Simon Lohrmann, Niklas Lüdtke, Esmée Oosterlaar

# Installation

(1) Create your Virtual Environment (here: conda)

    conda create -n DIL python=3.8.0
    conda activate DIL

> **Note:** We experienced some compatibility issues with later python versions. If possible, use python 3.8.0.


(2) Install all needed python packages using

    pip install -r requirements.txt

(3) Download the spacy models using the code shown below

    python -m spacy download en_core_web_sm
    python -m spacy download de_core_news_lg
    python -m spacy download en_core_web_trf


# Usage of Pipeline

To run the pipeline for a single PDF-file follow the steps below

1. Place PDF-file into "data" folder, named "file.pdf".

2. Run 
    ```
    python run.py
    ```

# Notes

If you want to use a different location or filename use the flag

```
python run.py --file_path=path/to/file.pdf
```

You can also specify a path for a different SelfORE model using

    python run.py --model_path=path/to/model.pdf

# Acknowledgements

This work is based significantly on [SelfORE](https://github.com/THU-BPM/SelfORE) (2020 Xuming Hu, Lijie Wen, Yusong Xu, Chenwei Zhang and Philip S. Yu). 
