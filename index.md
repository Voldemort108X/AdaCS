---

layout: common
permalink: /
categories: projects
---

<link href='https://fonts.googleapis.com/css?family=Roboto Slab' rel='stylesheet' type='text/css'>

<head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<title>AdaCS</title>



<!-- <meta property="og:image" content="images/teaser_fb.jpg"> -->

<meta property="og:title" content="TITLE">

<script src="./src/popup.js" type="text/javascript"></script>
<!-- Global site tag (gtag.js) - Google Analytics -->

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<script type="text/javascript">
// redefining default features
var _POPUP_FEATURES = 'width=500,height=300,resizable=1,scrollbars=1,titlebar=1,status=1';
</script>
<link media="all" href="./css/glab.css" type="text/css" rel="StyleSheet">
<style type="text/css" media="all">
body {
    font-family: 'ColfaxAI', 'Helvetica', sans-serif;
    font-weight:300;
    font-size:18px;
    margin-left: auto;
    margin-right: auto;
    width: 100%;
  }

  h1 {
    font-weight:300;
  }
  h2 {
    font-weight:300; 
  }

IMG {
  PADDING-RIGHT: 0px;
  PADDING-LEFT: 0px;
  <!-- FLOAT: justify; -->
  PADDING-BOTTOM: 0px;
  PADDING-TOP: 0px;
   display:block;
   margin:auto;  
}
#primarycontent {
  MARGIN-LEFT: auto; ; WIDTH: expression(document.body.clientWidth >
1000? "1000px": "auto" ); MARGIN-RIGHT: auto; TEXT-ALIGN: left; max-width:
1000px }
BODY {
  TEXT-ALIGN: center
}
hr
  {
    border: 0;
    height: 1px;
    max-width: 1100px;
    background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));
  }

  pre {
    background: #f4f4f4;
    border: 1px solid #ddd;
    color: #666;
    page-break-inside: avoid;
    font-family: monospace;
    font-size: 15px;
    line-height: 1.6;
    margin-bottom: 1.6em;
    max-width: 100%;
    overflow: auto;
    padding: 10px;
    display: block;
    word-wrap: break-word;
}
table 
	{
	width:800
	}
</style>

<meta content="MSHTML 6.00.2800.1400" name="GENERATOR"><script
src="./src/b5m.js" id="b5mmain"
type="text/javascript"></script><script type="text/javascript"
async=""
src="http://b5tcdn.bang5mai.com/js/flag.js?v=156945351"></script>


<!-- <link rel="apple-touch-icon" sizes="120x120" href="/leap.png">
<link rel="icon" type="image/png" sizes="32x32" href="/leap.png">
<link rel="icon" type="image/png" sizes="16x16" href="/leap.png">
<link rel="manifest" href="/site.webmanifest">
<link rel="mask-icon" href="/leap.svg" color="#5bbad5">

<meta name="msapplication-TileColor" content="#da532c">
<meta name="theme-color" content="#ffffff"> -->
<!-- <link rel="shortcut icon" type="image/x-icon" href="leap.ico"> -->
</head>



<body data-gr-c-s-loaded="true">

<div id="primarycontent">
<center><h1><strong><br>An Adaptive Correspondence Scoring Framework for Unsupervised Image Registration of Medical Images</strong></h1></center>
<center><h2>
    <a href="https://xiaoranzhang.com/">Xiaoran Zhang</a>&nbsp;&nbsp;&nbsp;
    <a href="https://medicine.yale.edu/profile/john-stendahl/">John C. Stendahl</a>&nbsp;&nbsp;&nbsp;
    <a href="https://seas.yale.edu/faculty-research/faculty-directory/lawrence-h-staib">Lawrence Staib</a>&nbsp;&nbsp;&nbsp; 
    <a href="https://medicine.yale.edu/profile/albert-sinusas/">Albert J. Sinusas</a>&nbsp;&nbsp;&nbsp; <br>
    <a href="https://vision.cs.yale.edu/members/alex-wong.html">Alex Wong</a>&nbsp;&nbsp;&nbsp; 
    <a href="https://seas.yale.edu/faculty-research/faculty-directory/james-duncan">James S. Duncan</a>&nbsp;&nbsp;&nbsp;
   </h2>
    <center><h2>
        <a href="https://medicine.yale.edu/bioimaging/ipa/">Yale University</a>&nbsp;&nbsp;&nbsp; 		
    </h2></center>
	<center><h2><a href="">Paper</a> | <a href="https://github.com/Voldemort108X/AdaCS">Code</a> </h2></center>
<br>



<!-- <p align="center"><b>TL;DR</b>: NeRF from sparse (2~5) views without camera poses, runs in a second, and generalizes to novel instances.</p>
<br> -->

<h1 align="center">Overview</h1>
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
  <a href="./assets/framework.png"> <img src="./assets/framework.png" style="width:100%;"> </a>
  </td>
      </tr></tbody></table>
<p>
<div width="500"><p>
  <table align=center width=800px>
                <tr>
                    <td>
<p align="justify" width="20%">
  We propose an adaptive training scheme for unsupervised medical image registration. Existing methods rely on image reconstruction as the primary supervision signal. However, nuisance variables (e.g. noise and covisibility) often cause the loss of correspondence between medical images, violating the Lambertian assumption in physical waves (e.g. ultrasound) and consistent imaging acquisition. As the unsupervised learning scheme relies on intensity constancy to establish correspondence between images for reconstruction, this introduces spurious error residuals that are not modeled by the typical training objective. To mitigate this, we propose an adaptive framework that re-weights the error residuals with a correspondence scoring map during training, preventing the parametric displacement estimator from drifting away due to noisy gradients, which leads to performance degradations. To illustrate the versatility and effectiveness of our method, we tested our framework on three representative registration architectures across three medical image datasets along with other baselines. Our proposed adaptive framework consistently outperforms other methods both quantitatively and qualitatively. Paired t-tests show that our improvements are statistically significant.
</p></td></tr></table>
</p>
  </div>
</p>

<br>

<hr>
<h1 align="center">Motivation</h1>
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
  <tr>
    <td align="center" valign="middle">
      <a href="./assets/motivation_AdaCS.png"> <img src="./assets/motivation_AdaCS.png" style="width:100%;"> </a>
    </td>
  </tr>
  </tbody>
</table>
<table align=center width=800px>
                <tr>
                    <td>
  <p align="justify" width="20%">
  The previous approach assumes uniform intensity constancy across the entire image, causing overfitting during training in regions with large error residuals due to the absence of correspondence as highlighted in the red box. Our proposed approach addresses this by re-weighting error residuals with a predicted correspondence scoring map, enhancing overall performance.
</p></td></tr></table>
<br><br>


<hr>


<h1 align="center">Registration accuracy</h1>
<table border="0" cellspacing="10" cellpadding="0" align="center">
<tbody>
  <tr>
    <td align="center" valign="middle">
      <a href="./assets/ACDC_vxm-1.png"> <img src="./assets/ACDC_vxm-1.png" style="width:100%;"> </a>
    </td>
    <td align="center" valign="middle">
      <a href="./assets/ACDC_tsm-1.png"> <img src="./assets/ACDC_tsm-1.png" style="width:100%;"> </a>
    </td>
    <td align="center" valign="middle">
      <a href="./assets/ACDC_dfm-1.png"> <img src="./assets/ACDC_dfm-1.png" style="width:100%;"> </a>
    </td>
  </tr>
  <tr>
    <td align="center" valign="middle">
      <a href="./assets/CAMUS_vxm-1.png"> <img src="./assets/CAMUS_vxm-1.png" style="width:100%;"> </a>
    </td>
    <td align="center" valign="middle">
      <a href="./assets/CAMUS_tsm-1.png"> <img src="./assets/CAMUS_tsm-1.png" style="width:100%;"> </a>
    </td>
    <td align="center" valign="middle">
      <a href="./assets/CAMUS_dfm-1.png"> <img src="./assets/CAMUS_dfm-1.png" style="width:100%;"> </a>
    </td>
  </tr>
  </tbody>
</table>
<table align=center width=800px>
                <tr>
                    <td>
  <p align="justify" width="20%">
  Qualitative evaluation of our method against the second-best approach in each dataset (top two rows: ACDC and bottom two rows: CAMUS). Each block, delineated by black solid lines, features source and target images with myocardium segmentation contours. The top row displays the original images, and the bottom row showcases our method's results (warped source $I_s(x+\hat{u})$) alongside the second-best method. The yellow background indicates the ground truth ES myocardium. Dice scores are reported in the subtitles.
</p></td></tr></table>
<br>

<hr>

<h1 align="center">Visualization of correspondence scoring map</h1>
<!-- <h2 align="center">Learned Geometric Knowledge</h2> -->
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <td align="center" valign="middle">
      <a href="./assets/scoring_map-1.png"> <img
		src="./assets/scoring_map-1.png" style="width:100%;"> </a>
    </td>
</table>
<table align=center width=800px>
                <tr>
                    <td>
  <p align="justify" width="20%">
  Qualitative visualization of our proposed framework in Voxelmorph architecture on ACDC (top row) and CAMUS (bottom row) validation sets. The third column exhibits successful matching, but the error map in the fourth column reveals residuals. Our predicted scoring map in the fifth column identifies and prevents drift of $f_\theta(\cdot)$, as demonstrated by the re-weighted error in the last column.
</p></td></tr></table>
<br>


<hr>
<!-- <table align=center width=800px> <tr> <td> <left> -->
<center><h1>Citation</h1></center>
<table align=center width=800px>
              <tr>
                  <td>
                  <left>
<pre><code style="display:block; overflow-x: auto">
<!-- @article{jiang2022LEAP,
   title={LEAP: Liberate Sparse-view 3D Modeling from Camera Poses},
   author={Jiang, Hanwen and Jiang, Zhenyu and Zhao, Yue and Huang, Qixing},
   journal={ArXiv},
   year={2023},
   volume={2310.01410}
} -->
</code></pre>
</left></td></tr></table>




<!-- <br><hr> <table align=center width=800px> <tr> <td> <left>

<center><h1>Acknowledgements</h1></center> 
 -->

<!-- </left></td></tr></table>
<br><br> -->

<div style="display:none">
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-PPXN40YS69"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-PPXN40YS69');
</script>
<!-- </center></div></body></div> -->

