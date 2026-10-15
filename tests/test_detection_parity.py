"""Backend parity matches sets without changing actual COCO score rankings."""
import unittest
from model_platform.algorithm import TrainingFailure
from model_platform.detection_training import _box_parity


def box(offset=0,score=.1,label=0):
    return {'class_id':label,'score':score,'box':[10+offset,10,20+offset,20]}


class DetectionParityTests(unittest.TestCase):
    def test_empty_exact_and_reordered_boxes(self):
        self.assertEqual(_box_parity([],[]),(0.,0.))
        self.assertEqual(_box_parity([box(),box(20)],[box(20),box()]),(0.,0.))
        error,score_error=_box_parity([box(score=.100001),box(20,score=.100002)],
                                    [box(20,score=.100001),box(score=.100002)])
        self.assertEqual(error,0);self.assertLessEqual(score_error,1e-4)

    def test_complete_matching_avoids_greedy_false_rejection(self):
        error,_=_box_parity([box(0),box(.09)],[box(.04),box(-.08)])
        self.assertLessEqual(error,.1)

    def test_no_actual_box_can_be_reused(self):
        with self.assertRaises(TrainingFailure):
            _box_parity([box(),box(.01)],[box(),box(20)])

    def test_count_class_score_and_coordinate_failures(self):
        for reference,actual in [([box()],[]),([box()],[box(label=1)]),
                                  ([box()],[box(score=.100101)]),([box()],[box(.10001)])]:
            with self.subTest(actual=actual),self.assertRaises(TrainingFailure):
                _box_parity(reference,actual)

    def test_input_order_and_values_not_mutated(self):
        import copy
        reference=[box(),box(20)];actual=list(reversed(reference));snapshot=copy.deepcopy((reference,actual))
        _box_parity(reference,actual)
        self.assertEqual((reference,actual),snapshot)


if __name__=='__main__':unittest.main()
