from pixeldebug.dataset import split_for, template, CATEGORIES, union_box
from pixeldebug.evaluate import iou


def test_family_split_is_deterministic_and_complete():
    splits = {split_for(i) for i in range(100)}
    assert splits == {'train', 'validation', 'test'}
    assert all(split_for(i) == split_for(i) for i in range(100))


def test_templates_and_boxes():
    assert len(CATEGORIES) == 6
    assert 'visibility:hidden' in template(1, 'missing_element')
    box = union_box(dict(x=0,y=0,width=10,height=10), dict(x=5,y=5,width=10,height=10))
    assert box == dict(x=0,y=0,width=15,height=15)
    assert iou(box, box) == 1
