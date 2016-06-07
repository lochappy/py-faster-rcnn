# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
from datasets.voc_eval import voc_ap
class inria(imdb):
    def __init__(self, image_set, devkit_path):
        imdb.__init__(self,"INRIA_Person_" + image_set)
        self._image_set = image_set
        self._devkit_path = devkit_path
        if image_set == 'train':
            self._data_path = os.path.join(self._devkit_path, 'data')
        else:
            self._data_path = os.path.join(self._devkit_path, 'testData')
        self._classes = ('__background__', # always index 0
                         'person')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = ['.jpg', '.png']
        self._image_index = self._load_image_set_index()
        self._list_of_output_detection_results = {}

        # Specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000,
                       'use_diff' : False,
                       'rpn_file' : None}

        assert os.path.exists(self._devkit_path), \
                'Devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        for ext in self._image_ext:
            image_path = os.path.join(self._data_path, 'Images',
                                  index + ext)
            if os.path.exists(image_path):
                break
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
	return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._data_path + /ImageSets/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_inria_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def rpn_roidb(self):
        gt_roidb = self.gt_roidb()
        rpn_roidb = self._load_rpn_roidb(gt_roidb)
        roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        #roidb = self._load_rpn_roidb(None)
        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_inria_annotation(self, index):
        """
        Load image and bounding boxes info from txt files of INRIAPerson.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.txt')
        # print 'Loading: {}'.format(filename)
    	with open(filename) as f:
                data = f.read()
    	import re
    	objs = re.findall('\(\d+, \d+\)[\s\-]+\(\d+, \d+\)', data)

        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        
        

        # "Seg" area here is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
	    coor = re.findall('\d+', obj)
            x1 = float(coor[0])
            y1 = float(coor[1])
            x2 = float(coor[2])
            y2 = float(coor[3])
            cls = self._class_to_ind['person']
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}
                
    def _get_path_of_inria_results_file(self, className):
        if className in self._list_of_output_detection_results:
            return self._list_of_output_detection_results[className]
        else:
            return None

    def _write_inria_results_file(self, all_boxes):
        
        use_salt = self.config['use_salt']
        comp_id = 'comp4'
        if use_salt:
            comp_id += '-{}'.format(os.getpid())

        # VOCdevkit/results/comp4-44503_det_test_aeroplane.txt
        if not os.path.exists(os.path.join(self._devkit_path, 'results')):
            os.mkdir(os.path.join(self._devkit_path, 'results'))
        if os.path.exists(os.path.join(self._devkit_path, 'results', self.name)):
            import shutil            
            shutil.rmtree(os.path.join(self._devkit_path, 'results', self.name))
        os.mkdir(os.path.join(self._devkit_path, 'results', self.name))
        path = os.path.join(self._devkit_path, 'results', self.name, comp_id + '_')
        self._list_of_output_detection_results = {}
        
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} results file'.format(cls)
            filename = path + 'det_'+ self._image_set + '_' + cls + '.txt'
            self._list_of_output_detection_results[cls] = filename
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
        return comp_id
        
    def _do_python_eval(self, output_dir = 'output'):
        
        use_07_metric = False
        aps = []
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
            
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            rec, prec, ap = self._voc_eval(cls, ovthresh=0.5, use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')
        
        
    def _voc_eval(self,classname,ovthresh=0.5, use_07_metric=False):
        # extract gt objects for this class
        class_recs = {}
        npos = 0
        imagenames = self.image_index
        clsIdx = self._class_to_ind[classname]
        for imagename in imagenames:
            ann = self._load_inria_annotation(imagename)
            gt_classes = ann['gt_classes']
            boxes = ann['boxes']
            bbox = np.array([boxes[i] for i in xrange(len(gt_classes)) if gt_classes[i] == clsIdx ]).astype(np.float)
            difficult = np.array([False]*bbox.shape[0])
            det = [False] * bbox.shape[0]
            npos = npos + bbox.shape[0]
            class_recs[imagename] = {'bbox': bbox,
                                     'difficult': difficult,
                                     'det': det}
                                     
        # read dets
        detfile = self._get_path_of_inria_results_file(classname)
        assert detfile != None, 'There is no detection output...'
        with open(detfile, 'r') as f:
            lines = f.readlines()
        
        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
        
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]
        
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]
        
        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih
                
                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                           (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                           (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                
            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.
                
        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
        
        return rec, prec, ap
        
    def evaluate_detections(self, all_boxes, output_dir):
        self._write_inria_results_file(all_boxes)
        self._do_python_eval(output_dir)
#
#
#
#
#testDb = inria('test','/home/lochappy/lochappy/00_detection_cnn/py-faster-rcnn/data/INRIA_Person_devkit')
#
##testDb._voc_eval('person')
#
##testDb._load_inria_annotation(testDb.image_index[0])
#
#with open('/home/lochappy/lochappy/00_detection_cnn/py-faster-rcnn/output/faster_rcnn_alt_opt/INRIA_Person_test/VGG_CNN_M_1024_faster_rcnn_final/detections.pkl', 'rb') as f:
#    all_boxes = cPickle.load(f)
#len(all_boxes[1][0].shape)
#
#testDb.evaluate_detections(all_boxes,'output')
#
#def inria_vis_detections(im, class_name, dets, thresh=0.3):
#    """Visual debugging of detections."""
#    import matplotlib.pyplot as plt
#    im = im[:, :, (2, 1, 0)]
#    for i in xrange(np.minimum(10, dets.shape[0])):
#        bbox = dets[i, :4]
#        score = dets[i, -1]
#        if score > thresh:
#            plt.cla()
#            plt.imshow(im)
#            plt.gca().add_patch(
#                plt.Rectangle((bbox[0], bbox[1]),
#                              bbox[2] - bbox[0],
#                              bbox[3] - bbox[1], fill=False,
#                              edgecolor='g', linewidth=3)
#                )
#            plt.title('{}  {:.3f}'.format(class_name, score))
#            plt.show()
#
#import cv2
#i=0
#im = cv2.imread(testDb.image_path_at(i))
#testDb._write_inria_results_file(all_boxes)
#inria_vis_detections(im,'Person',all_boxes[1][i],0.3)
#
#ovthresh = 0.5
#classname = 'person'







