<h1>Stolt Migration Deep-Learning</h1>
<p>This is a Deep Learning-based Frequency-Wavenumber (F-K) Migration pipeline for Ultrasound Image Reconstruction.</p>
<p>It replaces the traditional mathematical Stolt interpolation mapping step with a PyTorch Neural Network to process raw RF data into focused B-mode images.</p>

<h3>Description</h3>
<ul>
    <li><strong>Data_Loader.py</strong>: Responsible for loading in the complex ultrasound data from MATLAB (.mat) files and preparing it for PyTorch training.</li>
    <li><strong>dl_network.py</strong>: Holds the Neural Network architectures (e.g., SimpleUNet) to use for training.</li>
    <li><strong>trainer.py</strong>: The actual trainer used to train the neural network.</li>
    <li><strong>dl_fk_multi.py</strong>: The core processing pipeline that performs IQ demodulation, 3D FFTs, Deep Learning inference, and 2D IFFTs.</li>
    <li><strong>load_fk_model.py</strong>: A helper script that handles loading the trained PyTorch model weights and running complex-number inference.</li>
    <li><strong>validate_model.py</strong>: A validation script to process raw data, run the full pipeline, and visualize the final ultrasound image.</li>
    <li><strong>ptTargets_fk_multi_acq.m</strong>: The MATLAB script used to generate simulated multistatic data (via Field II) and create the required cache files.</li>
    <li><strong>/utils/Loss.py</strong>: Contains custom loss functions (like Pearson Correlation) used during training.</li>
    <li><strong>/trained_models</strong>: The directory where the trained model weights (e.g., <code>model_fk.pth</code>) are saved.</li>
    <li><strong>/cached_data</strong>: The directory where the MATLAB datasets (<code>beams_cached.mat</code>, <code>training_cache_fk.mat</code>) should be stored. Generate cached data via <strong>ptTargets_fk_multi_acq.m</strong></li>
</ul>

<h2>How to use</h2>
<h3>Data Generation</h3>
<ol>
    <li>Make sure you have MATLAB and the Field II simulator configured. (May require external sources)</li>
    <li>Run the data generation script: <code>ptTargets_fk_multi_acq.m</code> in MATLAB.</li>
    <li>This will simulate the data and automatically save <code>beams_cached.mat</code> and <code>training_cache_fk.mat</code> into the <code>/cached_data</code> directory.</li>
</ol>
<h3>Training</h3>
<ol>
    <li>Make sure you have Python installed along with the required libraries (<code>pip install torch scipy numpy matplotlib tqdm</code>).</li>
    <li>(Optional) Change configs and hyperparameters in <code>trainer.py</code>. You may add models / loss functions to the respective files and use them instead of the default settings.</li>
    <li>Run the trainer: <code>python trainer.py</code></li>
    <li>The trained model weights will be saved in <code>/trained_models</code>.</li>
</ol>
<h3>Testing & Validation</h3>
<ol>
    <li>Ensure the trained model (<code>model_fk.pth</code>) is present in the <code>/trained_models</code> directory.</li> 
    <li>Run the validation script: <code>python validate_model.py</code></li>
    <li>A Matplotlib window will open displaying the reconstructed ultrasound B-mode image using the trained network.</li>
</ol>

<h2>Acknowledgments</h2>
<p>This project is proprietary and all rights are reserved by the author. No permission is granted to copy, distribute, or modify the core deep learning architecture or training scripts.</p>

<p>Special thanks and credit to the following third-party project for its foundational code and mathematics:</p>
<ul>
    <li>
        <strong><a href="https://gitlab.com/mj66/bentobox">BentoBox</a></strong> (by Marko Jakovljevic & Louise Zhuang) - 
        The ultrasound simulation and dataset generation framework in <code>ptTargets_fk_multi_acq.m</code>, as well as the core signal processing pipeline (IQ demodulation, zero-padding, and 3D FFT) in <code>dl_fk_multi.py</code>, are direct Python/MATLAB translations and derivative works of the scripts found in their repository. The analytical Stolt mapping interpolation has been substituted in this project with a custom Deep Learning inference pipeline.
    </li>
</ul>
<p><em>Note: Third-party notices and original Stanford University liability disclaimers for the adapted algorithms are preserved in the <code>THIRD-PARTY-NOTICES.md</code> file.</em></p>