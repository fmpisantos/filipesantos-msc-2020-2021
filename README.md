How to run: 

0. Install python dependencies (only on the first run):
    ```
    git submodule update --init
    python3 -m venv .
    source bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
1. Start nima docker image (https://github.com/idealo/image-quality-assessment#serving-nima-with-tensorflow-serving):

    0. To reset docker containers (only if needed):
        
        ```Stop all running containers:``` 

            ```
            docker stop $(docker ps -a -q)
            ```

        ```Delete all stopped containers:```
        
            ```
            docker rm $(docker ps -a -q)
            ```

    1. Build the NIMA TFS Docker image 
        ```
        docker build -t tfs_nima iqa/contrib/tf_serving
        ```

    2. Run a NIMA TFS container with 
        ```
        docker run -d --name tfs_nima -p 8500:8500 tfs_nima
        ```

    3. run.sh is a script that runs all this steps from 1.0 to 1.2

2. TF_CPP_MIN_LOG_LEVEL=3 python main.py --output True
    ```
    optional arguments:
    -h, --help            show this help message and exit
    -fps FPS, --fps FPS   Frames per sec (default = 15).
    -is IMGSEC, --imgSec IMGSEC
                            Seconds per image (default = 1).
    -ts TSEC, --tSec TSEC
                            Seconds per transiction (default = 0.5).
    -ni NIMAGES, --nImage NIMAGES
                            Number of images to show (default = 15).
    -a ALG, --alg ALG     Select the algoritms to use from the following list (note: this flag can be omitted to use the recomended algoritms): (default = [] -> Runs all algoritms)
                                "brisque" or "b" for tecnical photo assessment
                                "nima" or "n" for aesthetic photo assessment
                                "labels" or "l" for image label identification
                                "objects" or "o" for objects identification
                                "identical" or "ii" for identical photos identification
                                "slideshow" or "s" to create a slideshow
    -p PATH, --path PATH  Path to folder holding the photos (default = './Photos/original').
    ```
3. Example:

    Default:

    ```
    TF_CPP_MIN_LOG_LEVEL=3 python main.py --output True
    ```

    With extra parameters:

    ```
    TF_CPP_MIN_LOG_LEVEL=3 python main.py -fps 10 -is 1.5 -ts 0 -ni 20 -a "brisque nima labels objects slideshow"
    ```

4. Known errors and solutions:
    1. Nima(iqa) docker image not running:
        ```
        Reason: 'TypeError("cannot pickle '_thread.RLock' object")'
        ```
    2. Multithreading image labeling not working on windows:
        ```
        ReferenceError: weakly-referenced object no longer exists
        ```
    3. Running dos2unix ./run.sh:
        ```
        unable to prepare context: path "iqa/contrib/tf_serving\r" not found
        ```
