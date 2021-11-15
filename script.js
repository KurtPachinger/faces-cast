let size, RAD;
let cOutput, app, stage, foreground, ploader, image, f, d, mousex, mousey;

let sto = {
  var: {},
  src: document.getElementById('chromaImg'),
  roi: document.getElementById('roi'),
  mask: document.getElementById('mask'),
  PYR: 0.5,
  batch: 0,
  swatch: {},
  orient: function(file) {
    sto.src.parentElement.removeChild(sto.src);

    loadImage(file, (img, data) => {
      console.log(img, data);
      img.id = 'chromaImg';
      img.setAttribute('data-batch', sto.batch);
      document.getElementById('edit').append(img);
      sto.src = img;
      sto.update(data);
    }, {
      orientation: true,
      canvas: true,
      crossOrigin: true,
      //minWidth: 256,
      maxWidth: document.getElementById('maxsize').value
    });

    let clear = document.getElementsByTagName('canvas');
    for (let i = 0; i < clear.length; i++) {
      let ctx = clear[i].getContext('2d');
      ctx && ctx.clearRect(0, 0, clear[i].width, clear[i].height);
    }

  },
  update: async function(complete, loading) {
    await new Promise((resolve) => {
      document.documentElement.classList.add('load');
      sto.updated = setTimeout(function() {
        clearTimeout(sto.updated);
        if (typeof complete === 'object') {
          //await image load, orientation, etc...
          if (sto.src.getAttribute('data-batch') == sto.batch) {
            sto.update();
          } else {
            sto.update(complete);
          }
          complete = 'load image';
        } else if (typeof complete === 'string') {
          console.timeLog('time');
        } else {
          if (!complete) {
            console.clear();
            console.time('time');
            document.getElementById('photo').disabled = true;
            sto.batch++;
            sto.swatch = {};
            sto.roi.width = sto.src.width;
            sto.roi.height = sto.src.height;
            cast.init();
          } else {
            // complete truthy 1 for no face path
            // abort checkbox will compile
            if (complete === 1) {
              document.getElementById('color').height = 0;
              document.getElementById('depth').height = 0;
              document.getElementById('chroma').height = 0;
            }
            cast.composite(complete);
            Pixi(complete);

            console.timeEnd('time');
            document.documentElement.classList.remove('load');
            document.getElementById('photo').disabled = false;
            document.getElementById('abort').checked = false;
            // remove old mask, chart/listener
            let old = sto.mask.querySelector('[data-batch=i' + (sto.batch - 1) + ']');
            if (old != null) {
              let bar = old.querySelectorAll('table.bar');
              for (let i = 0; i < bar.length; i++) {
                old[i] = null;
              }
              sto.mask.removeChild(old);
            }
            sto.mask.querySelector('[data-batch=i' + sto.batch + ']').classList.add('taint');
            // remove old mat
            for (let key in sto.swatch) {
              sto.delete(sto.swatch[key].mat);
            }

          }
          complete = 'setup';

        }

        chart(complete);

        document.querySelector('section')
          .setAttribute('data-load', complete);
        resolve(complete);

      }, 10);
      return complete;
    });

  },
  resize: function(obj, prc, centered) {
    if (obj.constructor.name === "Mat") {
      // if !prc, Mat size equals source (strict)
      let dec = (prc) ? new cv.Size(obj.cols * prc, obj.rows * prc) : size;
      cv.resize(obj, obj, dec, 0, 0, cv.INTER_NEAREST);
    } else {
      // if !centered, Rect x,y from origin (Mat space)
      let W = obj.width,
        H = obj.height;
      obj.width *= prc;
      obj.height *= prc;
      if (!centered) {
        obj.x *= prc;
        obj.y *= prc;
      } else {
        obj.x += (W - obj.width) / 2;
        obj.y += (H - obj.height) / 2;
      }
      // non-fractional
      obj.x = Math.floor(obj.x);
      obj.y = Math.floor(obj.y);
      obj.width = Math.ceil(obj.width);
      obj.height = Math.ceil(obj.height);
    }
    return obj;
  },
  delete: function(mat) {
    mat.setTo([0, 0, 0, 0]);
    cv.resize(mat, mat, new cv.Size(1, 1), 0, 0, cv.INTER_NEAREST);
    mat.delete();
  },
};

let utils = new Utils('errorMessage');
utils.loadOpenCv(() => {
  //docs.opencv.org/trunk/d2/df0/tutorial_js_table_of_contents_imgproc.html
  sto.GC = {
    //answers.opencv.org/question/132163/grabcut-mask-values/
    BGD: new cv.Scalar(cv.GC_BGD),
    FGD: new cv.Scalar(cv.GC_FGD),
    PR_BGD: new cv.Scalar(cv.GC_PR_BGD),
    PR_FGD: new cv.Scalar(cv.GC_PR_FGD),
  };

  utils.createFileFromUrl(
    '/haarcascade_frontalface_default.xml',
    '//raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml',
    function() {
      utils.createFileFromUrl(
        '/haarcascade_eye.xml',
        '//raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml',
        function() {
          sto.orient('//images-na.ssl-images-amazon.com/images/I/61Jb9zMgMgL._SL1001_.jpg');
        }
      );
    }
  );

});

document.getElementById('photo').onchange = function(e) {
  if (!e.target.files.length) {
    return;
  }
  sto.orient(e.target.files[0]);
};

document.getElementById('col').onchange = function(e) {
  sto.mask.classList.remove('col0', 'col1', 'col2');
  sto.mask.classList.add('col' + e.target.value);
};

let cast = {
  init: async function() {

    sto.var.src = cv.imread('chromaImg'); // note: reads canvas original, image responsive
    size = sto.var.src.size();

    let gray = new cv.Mat();
    cv.cvtColor(sto.var.src, gray, cv.COLOR_RGBA2GRAY, 0);

    //haarCascades
    let faces = new cv.RectVector();
    let faceCascade = new cv.CascadeClassifier();
    let eyes = new cv.RectVector();
    let eyeCascade = new cv.CascadeClassifier();
    faceCascade.load('haarcascade_frontalface_default.xml');
    eyeCascade.load('haarcascade_eye.xml');

    let minSize = new cv.Size(size.width * 0.05, size.height * 0.05),
      maxSize = new cv.Size(size.width, size.height);

    //www.emgu.com/wiki/files/1.5.0.0/Help/html/e2278977-87ea-8fa9-b78e-0e52cfe6406a.htm
    faceCascade.detectMultiScale(
      gray,
      faces,
      1.05,
      12,
      cv.CASCADE_DO_CANNY_PRUNING,
      minSize,
      maxSize
    );

    if (faces.get(0) === undefined) {
      console.info('no face');
      let err = document.createElement('div');
      err.setAttribute('data-batch', 'i' + sto.batch);
      err.innerText = 'no face';
      sto.mask.appendChild(err);
      sto.update(1);
    } else {
      // composite recolor of zones
      console.info('faces', faces.size());

      function ltr(rectVector) {
        //note: rectVector.get(i) becomes faces[i]
        let ltr = [];
        for (let i = 0; i < rectVector.size(); i++) {
          ltr.push(rectVector.get(i));
        }
        //two-factor sort, prioritize by y on 1/3 grid
        ltr.sort(function(a, b) {
          function prc(y) {
            return +(y / size.height * 100 / 33).toFixed();
          }
          return prc(a.y + (a.height / 2)) - prc(b.y + (b.height / 2)) || a.x - b.x;
        });
        return ltr;
      }

      var faces_ltr = ltr(faces);

      // output
      sto.var.dst = sto.var.src.clone();
      // depth, highlights, teeth
      sto.var.bmp = new cv.Mat.zeros(size, cv.CV_8UC1);
      // differ and pyramid use hsv
      sto.var.hsv = new cv.Mat();
      cv.cvtColor(sto.var.src, sto.var.hsv, cv.COLOR_RGB2HSV, 0);

      let batch = document.createElement('div');
      batch.setAttribute('data-batch', 'i' + sto.batch);
      sto.mask.append(batch);

      let eigen = [];

      for (let i = 0; i < faces_ltr.length; ++i) {
        let face = faces_ltr[i];
        let roiGray = gray.roi(face);
        eigen.push(sto.var.src.roi(face));

        //face reference shade rule
        let col = +(((192 / faces.size()) * i).toFixed());
        cast.roi_ui(face, [col, (192 - col), 255, 255]);
        let hr = document.createElement('hr');
        hr.setAttribute('style', 'color:rgb(' + col + ',' + (192 - col) + ',255)');
        batch.append(hr);

        //face index
        let roi = new cv.Rect();
        roi.idx = face.idx = 'f' + i + '_';
        let idx = document.createElement('div');
        idx.setAttribute('data-idx', roi.idx);
        idx.classList = 'row';
        batch.append(idx);

        //relative unit
        RAD = (Math.round(face.width / 64) * 2) + 1;

        // layers composite in order to mask/colorize
        face.name = 'bg';
        face.depth = 96;
        await cast.segment(face);

        roi.width = face.width * 0.50;
        roi.height = face.height * 0.125;
        roi.x = face.x + (face.width - roi.width) / 2;
        roi.y = face.y + (face.height * 0.50);
        roi.name = 'skin';
        roi.depth = 192;
        await cast.palette(roi, 6);

        let inMat;

        roi.width = face.width * 0.50;
        roi.height = face.height * 0.40;
        roi.x = face.x + (face.width - roi.width) / 2;
        roi.y = face.y + (face.height * 0.66);
        // in mat?
        inMat = (roi.y + roi.height);
        roi.height -= (inMat > size.height) ? inMat - size.height : 0;
        roi.name = 'lips';
        roi.depth = 224;
        await cast.palette(roi, 3, face);

        roi.width = face.width * 0.33;
        roi.height = face.height * 0.25;
        roi.x = face.x + (face.width - roi.width) / 2;
        roi.y = face.y - roi.height;
        // in mat?
        inMat = roi.y;
        roi.height -= (inMat < 0) ? 0 - inMat : 0;
        roi.y -= (inMat < 0) ? inMat : 0;
        roi.name = 'hair';
        roi.depth = 128; //provides motion ground (middle)
        await cast.palette(roi, 6, face);

        roi.depth = 160;
        // detect eyes in face ROI
        eyeCascade.detectMultiScale(roiGray, eyes, 1.05, 5, cv.CASCADE_FIND_BIGGEST_OBJECT | cv.CASCADE_DO_ROUGH_SEARCH,
          new cv.Size(face.width / 4, face.height / 4),
          new cv.Size(face.width / 2, face.height / 2));
        console.info('eyes', eyes.size());

        var eyes_ltr = ltr(eyes);
				let hash = 0;
        for (let j = 0; j < eyes_ltr.length; ++j) {
          let eye = eyes_ltr[j];

          if (face.y + eye.y > face.y + face.height * 0.5) {
            continue;
          } // no eye below nose

          roi.width = eye.width;
          roi.height = eye.height;
          roi.x = face.x + eye.x;
          roi.y = face.y + eye.y;
          roi.name = 'eye_' + hash;
          await cast.palette(roi, 3);

          roi.width = eye.width * 0.33;
          roi.height = eye.height * 0.33;
          roi.x = face.x + eye.x + eye.width * 0.33;
          roi.y = face.y + eye.y + eye.height * 0.33;
          roi.name = 'iris_' + hash;
          await cast.palette(roi, 3);

					hash++;
        }

        sto.delete(roiGray);

      }

      cast.absdiff(eigen);

      // depth hints on nose/hair, teeth hole
      sto.swatch.bump = {
        depth: 224,
        mat: sto.var.bmp
      };
      // final output
      sto.update(true);

    }

    sto.delete(gray);
    //mask.delete();
    faceCascade.delete();
    faces.delete();

  },
  roi_ui: function(rect, color) {
    let roi = cv.imread('roi');
    r1 = new cv.Point(rect.x, rect.y);
    r2 = new cv.Point(rect.x + rect.width, rect.y + rect.height);
    cv.rectangle(roi, r1, r2, color, 1, 4);
    cv.imshow('roi', roi);
    sto.delete(roi);
  },
  layer: function(idx, name, mat) {
    name = idx + name;
		
    let curr = sto.mask.querySelector('[data-batch=i' + sto.batch + ']');
    let prev = sto.mask.querySelector('[data-batch=i' + (sto.batch - 1) + ']') || document;

    //console.log('layer', name, gc);
    if (curr.querySelector('.cell.' + name)) {
      let pyrmask = document.createElement('canvas');
      pyrmask.id = 'gc_' + name;
      pyrmask.classList.add('hover');
      curr.querySelector('.' + name).appendChild(pyrmask);
      cv.imshow(pyrmask.id, mat);
      return;
    }

    let zone = document.createElement('div');
    zone.classList.add('cell', name);

    let qikmask = document.createElement('canvas');
    qikmask.id = 'ir_' + name;
    zone.appendChild(qikmask);

    let title = document.createElement('label');
    title.setAttribute('data-id', name);
		if(name.includes('skin')){
			title.setAttribute('data-skintone', sto.swatch[name].skintone);
		}

    // specific/existing, or default
    let recolor = prev.querySelector('#' + name) || document.querySelector('#default input[data-id=' + name + ']') || document.querySelector('#default input[data-id=' + name.replace(/(^f\d+_)|(_\d+)/g, "") + ']');
    let hex = name.includes('bg') ? '#ffffff' : '#00ff00';
    if (recolor) {
      if (!name.includes('bg')) { // bg shared (1/2)
        hex = recolor.value;
      }
      let rem = prev.querySelector('#' + name);
      if (rem && prev) {
        let find = prev.querySelector('.' + name);
        find.parentNode.removeChild(find);
      }
    }

    let color = document.createElement('input');
    color.setAttribute('type', 'color');
    color.setAttribute('id', name);
    color.setAttribute('data-id', name);
    color.setAttribute('value', hex);
    if (name.includes('bg')) {
      color.disabled = true;
    } // bg shared (2/2)
		

    title.appendChild(color);
    zone.appendChild(title);
    curr.querySelector('[data-idx=' + idx + ']').appendChild(zone);
    cv.imshow(qikmask.id, mat);

    //stacked chart output
    let cluster = sto.swatch[name] && sto.swatch[name].kmeans;

    let table = document.createElement('table');
    table.classList.add("bar");
    let cols = '';
    if (cluster) {
      //cols += '<th>' + '<span>' + sto.swatch[rect.idx + name].count + '</span></th>';
      for (let k = 0; k < cluster.length; k++) {
        let d = cluster[k];
				
				let color = "#000";
        let hue = d.rgb.r +  d.rgb.g +  d.rgb.b;
        if (hue < 384) {
          color = "#fff";
        }
				
				let distr = ((d.count / sto.swatch[name].count) * 100).toFixed(3);
				distr = isFinite(distr) ? distr : "";
        cols += '<td width="' + distr + '%"';
        cols += 'style="background-color:rgb(' + d.rgb.r + ',' + d.rgb.g + ',' + d.rgb.b + ')"';
        cols += 'data-id="' + name + k + '">';
        cols += '<span style="color:'+color+';">' + d.count + '</span>';
				cols += '</td>';
      }
    }
    table.innerHTML = cols;
    zone.prepend(table);
    //highlight chart
    table.addEventListener('mouseover', highlight, false);
    table.addEventListener('mouseout', highlight);
  },
  segment: async function(face) {
    // note: a.k.a. chroma-key garbage matte

    if (document.getElementById('abort').checked) {
      return;
    }

    await sto.update(face.name);
    console.log('%c' + face.idx + ' segment', 'background:#0080ff;');

    face = sto.resize(face, sto.PYR);
    let tmp = sto.resize(sto.var.src.clone(), sto.PYR);
    cv.cvtColor(tmp, tmp, cv.COLOR_RGBA2RGB, 0);
    let mask = new cv.Mat.zeros(tmp.size(), cv.CV_8UC1);

    // image grid corners sample (fyi decimate)
    let kmeans = [];
    let grid = tmp.clone();
    const CELL = 9;
		const card = ['NW','NE','SW','SE'];
    cv.resize(grid, grid, new cv.Size(CELL, CELL), 0, 0, cv.INTER_NEAREST);
    for (let i = 0; i < CELL; i++) {
      if (i !== 0 && i !== CELL - 1) {
        continue;
      }
      for (let j = 0; j < CELL; j++) {
        if (j !== 0 && j !== CELL - 1) {
          continue;
        }
        let rgb = grid.ucharPtr(i, j);
        kmeans.push({
          //count: +(100 / CELL).toFixed(2),
					count: card[0],
          rgb: {
            r: rgb[0],
            g: rgb[1],
            b: rgb[2]
          }
        });
				card.shift();
      }
    }
    grid.delete()

    let C = new cv.Point(face.x + (face.width / 2),
      face.y + (face.height / 2));
    let D = (face.width + face.height) / 2 / 8;

    //BACKGROUND
    mask.setTo(sto.GC.BGD);
    //VIGNETTE
    cv.circle(mask, C, face.width * 1.33, sto.GC.PR_BGD, -1, 4, 0);
    cv.circle(mask, C, face.width * 1.0, sto.GC.PR_FGD, -1, 4, 0);
    //OBSCURE (GROUP)
    cv.line(mask, new cv.Point(C.x, 0), new cv.Point(C.x, mask.rows), sto.GC.PR_FGD, D, 4, 0); //vert
    cv.line(mask, new cv.Point(0, C.y), new cv.Point(mask.cols, C.y), sto.GC.PR_BGD, D, 4, 0); //horz
    cv.ellipse(mask, new cv.Point(C.x - face.width / 2, C.y + D), new cv.Size(face.width * 0.66, face.height * 1.0), 180, 90, -90, sto.GC.BGD, RAD / 4, 0);
    cv.ellipse(mask, new cv.Point(C.x + face.width / 2, C.y + D), new cv.Size(face.width * 0.66, face.height * 1.0), 0, -90, 90, sto.GC.BGD, RAD / 4, 0);
    //EDGE
    cv.rectangle(mask, new cv.Point(0, 0), new cv.Point(mask.cols, mask.rows), sto.GC.BGD, D / 2, 4, 0);
    //HEAD
    cv.line(mask, new cv.Point(C.x - face.width * 0.33, C.y), new cv.Point(C.x + face.width * 0.33, C.y), sto.GC.FGD, D, 4, 0); //ear
    cv.circle(mask, new cv.Point(C.x, C.y - face.height / 2), D * 2, sto.GC.FGD, -1, 4, 0);
    cv.ellipse(mask, C, new cv.Size(face.width * 0.33, face.height * 0.66), 0, -360, 0, sto.GC.FGD, -1, 0);

    let recolor = new cv.Rect(0, 0, mask.cols, mask.rows);
    let bgdModel = new cv.Mat();
    let fgdModel = new cv.Mat();

    try {
      cv.grabCut(tmp, mask, recolor, bgdModel, fgdModel, 4, cv.GC_INIT_WITH_MASK);
    } catch (err) {
      console.warn('no obvious fg/bg');
      //return;
    } finally {
      sto.delete(bgdModel);
      sto.delete(fgdModel);
      sto.delete(tmp);
    }

    //segment depth
    for (let i = 0; i < mask.rows; i++) {
      for (let j = 0; j < mask.cols; j++) {
        let prob = mask.ucharPtr(i, j);
        switch (prob[0]) {
          case 1:
            prob[0] = 255;
            break;
          case 2:
            prob[0] = 64;
            break;
          case 3:
            prob[0] = 128;
            break;
          default:
            prob[0] = 0;
        }
      }
    }

    cv.bitwise_not(mask, mask);

    face = sto.resize(face, 1 / sto.PYR);
    mask = sto.resize(mask);

    //relax mask
    cv.medianBlur(mask, mask, RAD * 3);
    sto.swatch[face.idx + face.name] = {
      count: face.width * face.height,
      kmeans: kmeans,
      depth: face.depth,
      mat: mask,
      face: face
      //to: key blocks composite of multiple faces
    };

    cast.layer(face.idx, face.name, mask);

    cast.pyramid(face);

    // nose depth
    C = new cv.Point(face.x + (face.width * 0.5), face.y + (face.height * 0.5));
    cv.line(sto.var.bmp,
      new cv.Point(C.x, C.y - (face.height * 0.125)),
      new cv.Point(C.x, C.y + (face.height * 0.1)),
      [255, 255, 255, 255], RAD * 3, 4);

  },
  palette: async function(rect, lim) {

    //note: rect is rounded in pyramid, post-grow for loop
    if (document.getElementById('abort').checked) {
      return;
    }
    await sto.update(rect.name);

    cast.roi_ui(rect, [0, 255, 255, 255]);

    let name = rect.name;
    sto.swatch[rect.idx + rect.name] = {
      count: 0,
      kmeans: [],
      depth: rect.depth
    };
    let swatch = sto.swatch[rect.idx + rect.name];
    let cluster = swatch.kmeans;

    let tmp = sto.resize(sto.var.src.clone(), sto.PYR);
    rect = sto.resize(rect, sto.PYR);
    //roi for KMEANS palette
    let mat = tmp.roi(rect);
    let sample = new cv.Mat(mat.rows * mat.cols, 3, cv.CV_32F);
    for (let y = 0; y < mat.rows; y++) {
      for (let x = 0; x < mat.cols; x++) {
        for (let z = 0; z < 3; z++) {
          sample.floatPtr(y + x * mat.rows)[z] = mat.ucharPtr(y, x)[z];
        }
      }
    }

    let labels = new cv.Mat();
    let centers = new cv.Mat();
    let criteria = new cv.TermCriteria(
      cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER,
      8,
      0.90
    );
    cv.kmeans(sample, lim, labels, criteria, 4, cv.KMEANS_PP_CENTERS, centers);

    //console.log(labels.size(), centers.size());
    for (let x = 0; x < labels.size().height; x++) {
      var cluster_idx = labels.intAt(x, 0);
      let redChan = centers.floatAt(cluster_idx, 0);
      let greenChan = centers.floatAt(cluster_idx, 1);
      let blueChan = centers.floatAt(cluster_idx, 2);
      //initial pointer
      if (cluster[cluster_idx] === undefined) {
        //hsv model
        let hsv = new cv.Mat(1, 1, cv.CV_8UC3);
        hsv.setTo([redChan, greenChan, blueChan, 255]);
        cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV);
        //for color comparisons
        cluster[cluster_idx] = {
          count: 0,
          rgb: {
            'r': Math.round(redChan),
            'g': Math.round(greenChan),
            'b': Math.round(blueChan)
          },
          hsv: {
            'h': hsv.ucharPtr(0, 0)[0],
            's': hsv.ucharPtr(0, 0)[1],
            'v': hsv.ucharPtr(0, 0)[2]
          }
        };
        hsv.delete();
        swatch.count = labels.size().height;
      }
      //increment cluster
      cluster[cluster_idx].count++;
    }

    labels.delete();
    centers.delete();

    tmp = sto.resize(tmp);
    rect = sto.resize(rect, 1 / sto.PYR);

    if (name === 'skin') {
      swatch.skintone = 1;
    }

    //sort cluster counts desc
    function sortNumber(a, b) {
      let sv = (a.skintone == 1) ? 's' : 'v';
      return a.hsv[sv] - b.hsv[sv];
    }

    cluster.sort(sortNumber);
    if (name === 'skin') {
			//sat = [0=>255] [white=>color]
			//val = [0=>255] [black=>color]
      // if skin midtone dark, sort by value
			let sat = 0;
			let val = 0;
			let lim = Math.round(cluster.length/2)
      for(var i = 0; i < lim; i++) {
       sat += cluster[i].hsv.s;
			 val += cluster[i].hsv.v;
      }
      sat = sat / lim;
			val = val / lim;
			
      if (sat > 128 || val < 96) {
        swatch.skintone = 0;
        cluster.sort(sortNumber);
      }
    }

    // cast of lips/hair, otherwise inrange
    diff = cast.diff(rect);
    let range = diff.mat;
    rect.fit = diff.fit;

    if (diff.fit === false || (diff.fit && name === 'hair')) {
      //pad and cull outliers, affects mask growth
      let pad = (name === 'lips' || name.includes('eye')) ? 16 : 48;
      let min = (name == 'skin') ? (sto.swatch[rect.idx + 'skin'].skintone * 64) : 0;
      let off = {
        lo: (cluster[0].hsv.v >= 32 + min) ? cluster[0].rgb : cluster[1].rgb,
        hi: cluster[cluster.length - 2].rgb
      };

      let hi = new cv.Mat(tmp.rows, tmp.cols, tmp.type(), [off.hi.r + pad, off.hi.g + pad, off.hi.b + pad, 255]);
      pad = (name === 'skin') ? pad : pad / 2;
      let lo = new cv.Mat(tmp.rows, tmp.cols, tmp.type(), [off.lo.r - pad, off.lo.g - pad, off.lo.b - pad, 0]);

      cv.inRange(tmp, lo, hi, range);

      lo.delete();
      hi.delete();
    }

    if (name != 'skin' && name != 'hair') {

      // grow and crop ROI to improve mask
      let crop = new cv.Mat.zeros(size, cv.CV_8UC1);
      rect = sto.resize(rect, 1.33, true);

      cv.rectangle(crop,
        new cv.Point(rect.x, rect.y),
        new cv.Point(rect.x + rect.width, rect.y + rect.height),
        [255, 255, 255, 255], -1, 4);
      cv.bitwise_and(range, crop, range);

      sto.delete(crop);
      //display

      cast.roi_ui(rect, [255, 0, 255, 255]);
    }

    swatch.mat = range; // deleted on complete
    cast.layer(rect.idx, name, range);

    sto.delete(tmp)
    sto.delete(mat);
    sto.delete(sample);

    let recolor = document.getElementById(rect.idx + name).value;
    swatch.to = hexToRGB(recolor);

    console.log(name, swatch);
    if (diff.fit !== 0) {
      cast.pyramid(rect);
    }
  },
  absdiff: function(arr) {
    if (arr.length < 2) {
      return false;
    }

    let ico = new cv.Mat(128, 128, cv.CV_8UC1);
    ico.setTo([0, 0, 0, 255]);
    const resize = ico.size();

    for (let i = 0; i < arr.length; i++) {
      cv.resize(arr[i], arr[i], resize, 0, 0, cv.INTER_NEAREST);
    }

    for (let i = 0; i < arr.length; i++) {
      let src1 = arr[i];
      if (i < arr.length - 1) {
        for (let j = 0; j < arr.length - i - 1; j++) {
          let src2 = arr[i + 1];
          let diff = new cv.Mat();
          cv.absdiff(src1, src2, diff);
          cv.cvtColor(diff, diff, cv.COLOR_BGR2GRAY);
          cv.absdiff(ico, diff, ico);
          diff.delete();
        }
      }
      src1.delete();
    }

    cv.imshow('eigen', ico);
    ico.delete();
  },
  diff: function(rect, face) {
    // returns false, true, or 0
    let diff = {
      mat: new cv.Mat(),
      fit: false
    };

    if (rect.name != 'lips' && rect.name != 'hair') {
      return diff;
    }

    let mat = sto.var.hsv.roi(rect);
    let color = new cv.Mat.zeros(size.height, size.width, 0);
    let mask = new cv.Mat.zeros(size.height, size.width, 0);

    //todo: use skin above midtone?
    let hsv_skin = sto.swatch[rect.idx + 'skin'].kmeans;
    hsv_skin = hsv_skin[Math.round(hsv_skin.length / 2)].hsv;

    let int = 2;
    // loop pixels bottom-to-top (mouth overlaps chin)
    for (let y = mat.rows; y > 0; y -= int) {
      for (let x = mat.cols; x > 0; x -= int) {
        //translate roi coord to src
        let C = new cv.Point(x + rect.x, y + rect.y);

        // gradient ramp from top and horizontal mirrored
        let dx = Math.round(255 - (Math.abs(x - (mat.cols / 2)) / mat.cols) * 255);
        let dy = Math.round(255 - ((y / mat.rows) * 255));
        let alpha = Math.round((dx + dy) / 2);
        if (alpha < 136) {
          continue;
        } // falloff chin/cheeks

        // hsv variance
        let hsv_cast = {
          h: mat.ucharPtr(y, x)[0],
          s: mat.ucharPtr(y, x)[1],
          v: mat.ucharPtr(y, x)[2]
        };

        let hue = {
          min: Math.min(hsv_skin.h, hsv_cast.h),
          max: Math.max(hsv_skin.h, hsv_cast.h)
        };
        hue = (hue.max - hue.min > 90) ? hue.max - 180 - hue.min : hue.max - hue.min;
        hue = Math.abs(hue);
        let sat = Math.abs(hsv_skin.s - hsv_cast.s);
        let val = Math.abs(hsv_skin.v - hsv_cast.v);
        let prc = +((hue + sat + val) / (180 + 255 + 255)).toFixed(4);
        let int_rad = int / (1 - prc);
        //console.log('hsv', hue, sat, val, 'prc', prc, 'int_rad', int_rad);

        //mark if similar enough hsv
        let bandpass = hsv_cast.s < 16 || hsv_cast.v < 16;
        if (!bandpass &&
          (hue > 8 || prc > 0.25) &&
          (sat > 32 || val > 32)
        ) {
          cv.circle(mask, C, int_rad, [255, 255, 255, 255], -1, 4, 0);
          // offset inverse stroke breaks continuity of small periphary clusters
          cv.circle(mask, C, int_rad / (1 - prc) * RAD, [0, 0, 0, 255], int_rad / 4, 8, 0); //15-20x?
        }

      }
    }

    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();
    let hull = new cv.MatVector();
    cv.findContours(mask, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE); //_EXTERNAL or _CCOMP
    //console.log('contours', contours.size());

    let area_roi = rect.width * rect.height;
    let cast = [];
    for (let i = 0; i < contours.size(); ++i) {
      let cnt = contours.get(i);
      let area = (cv.contourArea(cnt, false) / area_roi);
      // cull by size and location
      // minimum area raised post-sort to view candidates but limit success
      if (
        (rect.name === 'hair' && area > 0.020) ||
        (rect.name !== 'hair' && area > 0.001 && area < 0.66)) {
        let M = cv.moments(cnt, false);
        let cx = M.m10 / M.m00;
        let cy = M.m01 / M.m00;
        let off = {
          x: +(Math.abs((rect.x + (rect.width / 2) - cx) / rect.width * 100).toFixed(2)),
          y: +(Math.abs((rect.y + (rect.height / 2) - cy) / rect.height * 100).toFixed(2))
        };
        if ((off.x < 33 && off.y < 33) || rect.name == 'hair') {
          cast.push({
            idx: i,
            area: area,
            centroid: {
              x: cx,
              y: cy
            },
            off: off
          });
        }
      }
      let tmp = new cv.Mat();
      cv.convexHull(cnt, tmp, false, true);
      hull.push_back(tmp);
      tmp.delete();
      cnt.delete();
    }

    if (cast.length) {

      cast.sort(function(a, b) {
        function prc(x, axis) {
          axis = axis || 'width';
          return +(x / rect[axis] * 100 / 33).toFixed();
        }
        return b.area.toFixed(2) - a.area.toFixed(2) ||
          prc(b.off.x) - prc(a.off.x) ||
          prc(b.off.y, 'height') - prc(a.off.y, 'height');
      });

      const CUT = 96;
      for (let i = cast.length - 1; i >= 0; i--) {
        let pow = 255 - Math.round(CUT * (i / cast.length));
        //console.log(pow);
        cv.drawContours(color, hull, cast[i].idx, [pow - CUT, pow - CUT, pow - CUT, 255], -1, 8, hierarchy, 0);
        cv.drawContours(color, contours, cast[i].idx, [pow, pow, pow, 255], -1, 4, hierarchy, 0);

      }
    }
    diff.fit = (cast.length && cast[0].area > 0.005) ? true : 0;
    console.log(rect.name + ' cast', cast, diff.fit);
    if (!diff.fit) {
      console.warn(rect.name + ' no fit')
    }

    hierarchy.delete();
    contours.delete();
    sto.delete(mask);
    diff.mat = color;
    return diff;
  },
  pyramid: function(rect) {
    let name = rect.name;
    let tmp = sto.var.src.clone();
    cv.cvtColor(tmp, tmp, cv.COLOR_RGBA2RGB, 0);

    let mask = new cv.Mat.zeros(size, cv.CV_8UC1);
    mask.setTo(sto.GC.BGD);

    let bg_mat = sto.swatch[rect.idx + 'bg'].mat;
    let inrange_mat = (name != 'bg') ? sto.swatch[rect.idx + name].mat : bg_mat;

    let skin_mat = (name == 'lips' || name == 'hair') ? sto.swatch[rect.idx + 'skin'].mat : false;

    let hair_mat = new cv.Mat();
    if (name.includes('eye')) {
      //compare to relaxed hair
      let open = RAD * 2;
      hair_mat = sto.swatch[rect.idx + 'hair'].mat.clone();
      let M = cv.Mat.ones(open, open, cv.CV_8U);
      let anchor = new cv.Point(-1, -1);
      cv.morphologyEx(hair_mat, hair_mat, cv.MORPH_OPEN, M, anchor, 2,
        cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());
      cv.medianBlur(hair_mat, hair_mat, open + 1);
      M.delete();
    }

    // note: effectively an roi, and clips tiered output
    // todo: max all ~3x head size
    //console.warn(face)
    let all = rect;
    if (!(name.includes('eye') || name.includes('iris'))) {
      let face = sto.swatch[rect.idx + 'bg'].face;
      let pad = face.width;
      let X1 = face.x - pad;
      let Y1 = face.y - pad;
      let X2 = face.width * 3;
      let Y2 = face.height * 3;

      if (name == 'bg' || X1 < 0) {
        X1 = 0;
      }
      if (name == 'bg' || Y1 < 0) {
        Y1 = 0;
      }
      if (name == 'bg' || X2 > size.width) {
        X2 = size.width;
      }
      if (name == 'bg' || Y2 > size.height) {
        Y2 = size.height;
      }

      all = new cv.Rect(X1, Y1, X2, Y2);
    }

    let skintone = (name != 'bg') ? sto.swatch[rect.idx + 'skin'].skintone : null;
    let int = 2;
    for (let i = all.y; i < all.y + all.height; i += int) {
      for (let j = all.x; j < all.x + all.width; j += int) {

        let bg = bg_mat.ucharAt(i, j);
        let inrange = inrange_mat.ucharAt(i, j);
        let skin = skin_mat ? skin_mat.ucharAt(i, j) : 0;

        let C = new cv.Point(j, i);
        // refine masks in 2-factor waterfall

        if (name == 'bg') {
          bg = 255 - bg
        }
        if (bg <= 128) {
          if (inrange >= 128) {
            cv.circle(mask, C, int, sto.GC.FGD, -1, 4, 0);

            // throttle critical overlap
            if (name === 'hair') {
              if (skin >= 128) {
                cv.circle(mask, C, int, sto.GC.PR_BGD, -1, 4, 0);
              }
              if (sto.swatch[rect.idx + 'lips'].mat.ucharAt(i, j) >= 128) { //todo: maybe all?
                cv.circle(mask, C, int, sto.GC.BGD, -1, 4, 0);
              }
            } else if (name.includes('eye')) {
              if (hair_mat.ucharAt(i, j) >= 128) {
                cv.circle(mask, C, int, sto.GC.BGD, -1, 4, 0);
              }
            }

          } else {
            cv.circle(mask, C, int, sto.GC.PR_FGD, -1, 4, 0);
          }

          // throttle teeth and highlights (color=bg, depth=fg)
          // ROI for distant faces is larger, less depth detail
          if (name == 'lips') {
            //sat = [0=>255] [white=>color]
            let sat = sto.var.hsv.ucharPtr(i, j)[1];
            let val = sto.var.hsv.ucharPtr(i, j)[2];
            if (bg == 0 && skin >= 128) { // narrow head area
              cv.circle(mask, C, int, sto.GC.PR_FGD, -1, 4, 0);
              if (sat <= 64 - (32 * skintone)) { // broad flesh (highlights * skintone)
                cv.circle(sto.var.bmp, C, int, [255, 255, 255, 255], -1, 4, 0);
                if (sat <= 32) { // narrow teeth
                  cv.circle(mask, C, int, sto.GC.PR_BGD, -1, 4, 0);
                }

              }
            } else if (skin <= 128) { // broad hair area
              if (sat < 128 && val > 64) {
                cv.circle(sto.var.bmp, C, int, [255, 255, 255, 255], -1, 4, 0);
              }
            }
          }

        } else {
          if (inrange >= 128) {
            cv.circle(mask, C, int, sto.GC.PR_BGD, -1, 4, 0);
          } else {
            cv.circle(mask, C, int, sto.GC.BGD, -1, 4, 0);
          }
        }

      }
    }

    let bgdModel = new cv.Mat();
    let fgdModel = new cv.Mat();
    //greedy sample area (again)?

    try {
      cv.grabCut(tmp, mask, rect, bgdModel, fgdModel, 4, cv.GC_INIT_WITH_MASK);
    } catch (err) {
      console.warn('no obvious fg/bg');
      //return;
    } finally {
      sto.delete(bgdModel);
      sto.delete(fgdModel);
      hair_mat.delete();
      sto.delete(tmp);
    }

    // throttle alpha mask
    for (let i = 0; i < mask.rows; i++) {
      for (let j = 0; j < mask.cols; j++) {
        let pyr = mask.ucharPtr(i, j);
        switch (pyr[0]) {
          case 1:
            pyr[0] = 255;
            break;
          case 2:
            pyr[0] = 64;
            break;
          case 3:
            pyr[0] = 128;
            break;
          default:
            pyr[0] = 0;
        }
      }
    }

    // output refinement
    sto.swatch[rect.idx + name].mat = mask;
    cast.layer(rect.idx, name, mask);
    //recolor
    cv.threshold(mask, mask, 128, 255, cv.THRESH_BINARY);
    // relaxed eye area, reduced depth stretch
    if (name.includes('eye')) {
      let M = cv.Mat.ones(RAD * 2, RAD * 2, cv.CV_8U);
      cv.morphologyEx(mask, mask, cv.MORPH_CLOSE, M);
      M.delete();
      cv.medianBlur(mask, mask, RAD);
    }
    // target color
    let rgb = sto.swatch[rect.idx + name].to;

    cast.recolor(sto.var.dst, mask, rgb);

  },
  composite: function(complete) {

    //if face composite, clean globals
    if (complete !== 1) {
      // composite and focal blur depth of field
      let background = new cv.Mat.zeros(size, cv.CV_8UC1);
      // mask color multiply white
      let colors = new cv.Mat(size.width, size.height, cv.CV_8UC4);
      // mask depth moderate black
      let depth = new cv.Mat(size.width, size.height, cv.CV_8UC4, [64, 64, 64, 255]);
      let ksize = new cv.Size(RAD * 3, RAD * 3);

      for (let key in sto.swatch) {
        let layer = sto.swatch[key].mat.clone();
        let mask = new cv.Mat(size, cv.CV_8UC1);
        // mask binary
        cv.threshold(layer, mask, 128, 255, cv.THRESH_BINARY);
        // color/depth alpha
        cv.cvtColor(layer, layer, cv.COLOR_GRAY2RGBA);
        //color, chroma
        let rgb = sto.swatch[key].to;

        // depth layers
        let gray = sto.swatch[key].depth;
        gray = [gray, gray, gray, 255];
        layer.setTo(gray);
        // foreground depth (no holes)
        let inverse = new cv.Mat();
        cv.bitwise_not(mask, inverse);
        if (key.includes('bg')) {
          // dilate to reduce tearing
          let pad = new cv.Mat();
          let anchor = new cv.Point(-1, -1);
          let M = cv.Mat.ones(RAD * 5, RAD * 5, cv.CV_8U);
          cv.dilate(inverse, pad, M, anchor, 1, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());
          layer.copyTo(depth, pad);
          sto.delete(pad);
          // also background color (once inverted)
          layer.setTo([255, 255, 255, 255]);
          layer.copyTo(background, inverse);
        } else {
          // feature depth
          if (key !== 'bump') {
            layer.setTo(gray);
          }
          layer.copyTo(depth, mask);
        }

        // color
        if (rgb) {
          layer.setTo([rgb.r, rgb.g, rgb.b, rgb.a]);
          layer.copyTo(colors, mask);
        }

        sto.delete(inverse);
        sto.delete(mask);
        sto.delete(layer);
      }

      // depth output
      cv.GaussianBlur(depth, depth, ksize, RAD, RAD, cv.BORDER_DEFAULT);
      cv.imshow('depth', depth);
      sto.delete(depth);

      let rgb = document.querySelector('input[data-id=bg]').value;
      rgb = hexToRGB(rgb);
      console.log('bg', rgb);

      cv.bitwise_not(background, background);
      rgb.a && colors.setTo([rgb.r, rgb.g, rgb.b, rgb.a], background);
      cv.imshow('color', colors);
      sto.delete(colors);

      // output bg effects (blur, unsharp, color...?)
      rgb.a && cast.recolor(sto.var.src, background, rgb);
      cv.GaussianBlur(sto.var.src, sto.var.src, new cv.Size(RAD, RAD), 0, 0, cv.BORDER_DEFAULT);
      sto.var.src.copyTo(sto.var.dst, background);

      cv.imshow('chroma', sto.var.dst);

      sto.delete(background);

      sto.delete(sto.var.dst);
      sto.delete(sto.var.hsv);
    }
    sto.delete(sto.var.src);

  },
  recolor: function(mat, mask, col) {
    if (!col || col.a === 0) {
      return;
    }
    let tmp = sto.var.src.clone();
    cv.cvtColor(tmp, tmp, cv.COLOR_RGB2RGBA, 0);
    for (let i = 0; i < mat.rows; i++) {
      for (let j = 0; j < mat.cols; j++) {
        //stackoverflow.com/questions/5825149/
        let src = sto.var.src.ucharPtr(i, j);
        let mlt = tmp.ucharPtr(i, j);
        mlt[0] = (src[0] * col.r) / 255;
        mlt[1] = (src[1] * col.g) / 255;
        mlt[2] = (src[2] * col.b) / 255;
        //mask.ucharAt(i, j)[0] = col.a;
      }
    }
    tmp.copyTo(mat, mask);
    sto.delete(tmp);
  }

}

function chart(complete) {
  //codepen.io/pinstripe/pen/RVEEzX
  if (!!complete) {
    d3.select('#wheel').selectAll('g.Area,g.Center,use').remove();

  }
  if (Object.keys(sto.swatch).length === 0) {
    return;
  }

  let g;
  const angleOffset = 90;

  const toRadians = degrees => degrees * (Math.PI / 180);
  const toDegrees = radians => radians * (180 / Math.PI);
  const getPointAtAngle = (angle, len) => ({
    x: Math.cos(toRadians(angle - angleOffset)) * len,
    y: Math.sin(toRadians(angle - angleOffset)) * len
  });

  if (d3.select('#wheel svg').size() < 1) {
    const slices = 36;
    const spokes = [];
    while (spokes.length < slices) {
      spokes.push(spokes.length);
    }

    const angleScale = d3.scaleBand()
      .rangeRound([0, 360])
      .domain(spokes)
      .paddingOuter(0)
      .paddingInner(0);

    //render wheel
    let cfg = scheme();
    console.log('d3', cfg);

    const wheel = d3.select('#wheel')
      .append('svg')
      .attr('viewBox', `0 0 250 250`);

    wheel.append('defs')
      .selectAll('clipPath')
      .data(cfg.steps)
      .enter()
      .append('clipPath')
      .attr('id', (d, i) => `clip-${ i }`)
      .append('circle')
      .attr('r', d => d.r);

    g = wheel
      .append('g')
      .attr('transform', 'translate(125, 125)');

    g.append('g')
      .attr('class', '.Wheel__segmentGroups')
      .selectAll('.SegmentGroups')
      .data(cfg.steps)
      .enter()
      .append('g')
      .attr('class', 'SegmentGroup')
      .attr('clip-path', (d, i) => `url(#clip-${ i })`)
      .selectAll('.Segment')
      .data(spokes)
      .enter()
      .append('path')
      .attr('class', 'Segment')
      .attr('d', n => {
        const x0 = angleScale(n);
        const x1 = x0 + angleScale.bandwidth() + 1;
        const pt0 = getPointAtAngle(x0, 200);
        const pt1 = getPointAtAngle(x1, 200);
        return `M 0 0 L ${ pt0.x } ${ pt0.y } L ${ pt1.x } ${ pt1.y } L 0 0`;
      })
      .attr('fill', function(n) {
        const {
          i
        } = d3.select(this.parentNode).datum();
        return cfg.color[n][i];
      });

    const spokePoint = n => getPointAtAngle(angleScale(n) + angleScale.bandwidth() / 2 - (360 / slices / 2), cfg.steps[0].r + ((cfg.steps[1].r - cfg.steps[0].r) / 2));

    const spoke = g
      .append('g')
      .attr('class', 'Spokes')
      .selectAll('.Spoke')
      .data(spokes)
      .enter()
      .append('g')
      .attr('class', n => {
        let skip = 'Spoke ';
        skip += ((angleScale(n) % 60) === 0) ? '' : 'hide';
        return skip;
      })
      .attr('transform', n => {
        const {
          x,
          y
        } = spokePoint(n);
        return `translate(${ x*1.25 }, ${ y*1.25 })`;
      });

    spoke.append('circle')
      .attr('r', angleScale.bandwidth());

    spoke.append('text')
      .attr('dy', '0.5em')
      .text(angleScale);

    // origin
    g.append('line')
      .attr('x1', 0)
      .attr('x2', 0)
      .attr('y1', 0)
      .attr('y2', -100);

    // sort data OpenCV to d3
    function scheme() {
      const hsl = [];
      const steps = [];
      let div = 4;
      for (let n = 0; n < slices; n++) {
        hsl[n] = [];
        for (let j = 0; j < div; j++) {
          const h = angleScale(n) + angleScale.bandwidth() / 2;
          hsl[n].push(d3.hsl(h,
            1 - (j / div),
            0.5));
          if (n === 0 && steps.length < div) {
            steps.push({
              r: 100 - ((1 / div) * 100 * j),
              i: j
            });
          }
        }
      }

      return {
        color: hsl,
        steps: steps
      };
    }

    return;
  }

  // data from OpenCV to d3
  let data = [];
  for (let key in sto.swatch) {
    let kmeans = sto.swatch[key].kmeans;
    if (kmeans != undefined && !key.includes('bg')) {
      //console.log(key, kmeans);
      let zone = [];
      for (let j = 0; j < kmeans.length; j++) {
        let cluster = {};
        if (kmeans[j]) {
          cluster.zone = key;
          cluster.count = kmeans[j].count;
          cluster.x = kmeans[j].hsv.h * 2; //180=>360
          cluster.y = +(kmeans[j].hsv.s * (100 / 255)).toFixed(2); //255=>100
          cluster.rgb = kmeans[j].rgb;
          cluster.id = key + j; //SVG z-index render order
        }
        zone.push(cluster);
      }
      //cs.stackexchange.com/questions/52606/sort-a-list-of-points-to-form-a-non-self-intersecting-polygon
      data.push(zone);
    }
  }
  //console.log('d3', data);

  let total = (size.width + size.height) / 2;
  const areaScale = d3
    .scaleLinear()
    .domain([0, total / 2]) //input bounds
    .range([2, total / 8]); //output bounds

  //bl.ocks.org/d3indepth/b6d4845973089bc1012dec1674d3aff8 // todo: area
  // add kmeans centers to wheel
  g = d3.select('#wheel g');
  const area = g.append('g').attr('class', 'Area');
  const center = g.append('g').attr('class', 'Center');
  for (let j = 0; j < data.length; j++) {
    center
      .selectAll('circle.point')
      .data(data.flat())
      .enter()
      .append('circle')
      .attr('class', 'point')
      .attr('transform', (n, i) => {
        const {
          x,
          y
        } = getPointAtAngle(n.x, n.y);
        return `translate(${ x }, ${ y })`;
      })
      .attr('r', (d) => areaScale(Math.sqrt(d.count / Math.PI * 2)))
      .attr('fill', (d) => 'rgb(' + d.rgb.r + ',' + d.rgb.g + ',' + d.rgb.b + ')')
      .attr('id', (d) => d.id);
  }

  g
    .append('use')
    .attr('id', 'SVGorder')
    .attr('xlink:href', '#');

}

function highlight(e) {
  let target = e.target;
  if (target.tagName.toUpperCase() === 'TD') {
    let id = target.getAttribute('data-id');
    target = document.getElementById(id);
    if (target) {
      if (e.type === 'mouseover') {
        target.classList.add('active');
        document.getElementById('SVGorder')
          .setAttributeNS('http://www.w3.org/1999/xlink', 'xlink:href', '#' + id);
        document.getElementById('SVGorder')
          .setAttributeNS('http://www.w3.org/1999/xlink', 'transform', target.getAttribute('transform'));
      } else {
        target.classList.remove('active');
      }
    }
  }
}

function hexToRGB(h) {
  //css-tricks.com/converting-color-spaces-in-javascript/
  let r = 0,
    g = 0,
    b = 0;
  // 3 digits
  if (h.length === 4) {
    r = '0x' + h[1] + h[1];
    g = '0x' + h[2] + h[2];
    b = '0x' + h[3] + h[3];
    // 6 digits
  } else if (h.length === 7) {
    r = '0x' + h[1] + h[2];
    g = '0x' + h[3] + h[4];
    b = '0x' + h[5] + h[6];
  }
  // 256
  r = parseInt(r, 16);
  g = parseInt(g, 16);
  b = parseInt(b, 16);

  let a = (r === 0 && g === 255 && b === 0) ? 0 : 255;
  return {
    r: r,
    g: g,
    b: b,
    a: a
  };
}



function Pixi(complete) {
  console.log('PIXI', complete);
  let WEBGL = PIXI.utils.isWebGLSupported();
  if (!WEBGL) {
    //no driver, low resources... needs legacy canvas
    alert('no WEBGL = no PIXI depth map');
  }

  //if (Object.keys(sto.swatch).length === 0) {
    //return;
  //}
	
	//cleanup START
	// note: Pixi 5+ was giving me trouble with repeated instances, in terms of vertical position and memory.
	let els = document.querySelectorAll('#view .abs');
	for(let i=0;i<els.length;i++){
		let el = els[i];
	  el.parentNode.removeChild(el);
	}
	if(ploader!=undefined){
	for (let key in ploader.resources) {
    let tex = ploader.resources[key];
    tex && tex.texture && tex.texture.baseTexture.destroy();
    ploader.destroy(key);
  }
		ploader.reset();
	}
  if(foreground!=undefined){
  foreground.removeChildren();
	}
	//cleanup END


    cOutput = document.getElementById('view');
    app = new PIXI.autoDetectRenderer({
      width: size.width,
      height: size.height
    });
    cOutput.appendChild(app.view);
    app.view.classList.add('abs');
    app.view.id = 'inpaint_pixi';
    // asset pointer
    stage = new PIXI.Container();
    foreground = new PIXI.Container();
    stage.addChild(foreground);
    ploader = new PIXI.Loader();

    cOutput.onpointermove = function(e) {
      //parallax
      mousex = e.clientX;
      mousey = e.clientY;
    };
    animate();


  mousex = (cOutput.offsetWidth / 2) + cOutput.offsetLeft;
  mousey = (cOutput.offsetHeight / 2) + cOutput.offsetTop;
  //asset update
  ploader.add('image' + sto.batch, document.getElementById('chroma').toDataURL('image/jpeg', 1.0));
  ploader.add('depth' + sto.batch, document.getElementById('depth').toDataURL('image/jpeg', 0.5));
  ploader.onComplete.add(startMagic)
  ploader.load();

  function startMagic() {
    image = new PIXI.Sprite(ploader.resources['image' + sto.batch].texture);
    foreground.addChild(image);
    d = new PIXI.Sprite(ploader.resources['depth' + sto.batch].texture);
    f = new PIXI.filters.DisplacementFilter(d, 0);
    image.filters = [f];
  }

}

function animate() {
  if (!!f && !!d) {
    f.scale.x = (cOutput.offsetWidth / 2 - (mousex - cOutput.offsetLeft)) / 10;
    f.scale.y = (cOutput.offsetHeight / 2 - (mousey - cOutput.offsetTop)) / 10;
    image.addChild(d);
    d.renderable = false;
    app.render(stage);
  }
  requestAnimationFrame(animate);
}
