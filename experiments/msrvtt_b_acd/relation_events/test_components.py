"""Focused tests of the new scientific assumptions, runnable without an LLM."""
import unittest

import torch
from .association import associate, make_events, linear_sum_assignment
from .branch import (RelationEventBranch, eligible_order_caption,
                     order_ranking_loss, reverse_event_dynamics)


class AssociationTests(unittest.TestCase):
    def test_exact_assignment_matches_exhaustive_small_rectangles(self):
        import itertools
        for rows, columns in [(3, 3), (2, 4), (4, 2)]:
            torch.manual_seed(rows * 10 + columns)
            cost = torch.randn(rows, columns)
            left, right = linear_sum_assignment(cost.numpy())
            actual = sum(float(cost[a, b]) for a, b in zip(left, right))
            if rows <= columns:
                expected = min(sum(float(cost[i, j]) for i, j in enumerate(p))
                               for p in itertools.permutations(range(columns), rows))
            else:
                expected = min(sum(float(cost[i, j]) for j, i in enumerate(p))
                               for p in itertools.permutations(range(rows), columns))
            self.assertAlmostEqual(actual, expected, places=6)

    def inputs(self):
        categories = torch.ones(2, 2, dtype=torch.long)
        boxes = torch.tensor([[[0., 0., .2, .2], [.8, .8, 1., 1.]],
                              [[.78, .78, .98, .98], [.02, .02, .22, .22]]])
        geometry = torch.zeros(2, 2, 12); geometry[..., :4] = boxes
        appearance = torch.tensor([[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]])
        valid = torch.ones(2, 2, dtype=torch.bool)
        return categories, geometry, appearance, valid, torch.tensor([0., 1.]), torch.tensor([True, False])

    def test_detection_order_is_not_identity(self):
        previous, quality = associate(*self.inputs())
        self.assertEqual(previous[1].tolist(), [1, 0])
        self.assertTrue(bool((quality[1] > 0.5).all()))

    def test_no_links_across_shot_boundary(self):
        inputs = list(self.inputs()); inputs[-1] = torch.tensor([True, True])
        previous, quality = associate(*inputs)
        self.assertTrue(bool((previous == -1).all()))
        self.assertEqual(float(quality.sum()), 0.)

    def test_category_mismatch_rejected(self):
        inputs = list(self.inputs()); inputs[0][1] = 2
        previous, _ = associate(*inputs)
        self.assertTrue(bool((previous == -1).all()))

    def test_ambiguous_duplicates_get_low_reliability(self):
        inputs = list(self.inputs()); inputs[1][...] = 0
        inputs[1][..., 2:4] = 1
        inputs[2][...] = 1
        previous, quality = associate(*inputs)
        self.assertTrue(bool((previous[1] >= 0).all()))
        self.assertTrue(bool((quality[1] < .01).all()))

    def test_rate_uses_timestamp_interval_and_matched_pair(self):
        ids, geo, appearance, valid, times, shots = self.inputs()
        geo[..., 9] = .9
        relations = torch.zeros(2, 2, 2, 9)
        relations[0, 0, 1, 0] = .8
        relations[1, 1, 0, 0] = .4
        times[1] = 2.
        spatial = dict(object_ids=ids, object_geometry=geo, object_mask=valid,
                       object_relations=relations, timestamps=times,
                       frame_mask=torch.ones(2, dtype=torch.bool))
        events = make_events(spatial, appearance, shots, maximum_events=2)
        # Current pair (1,0) corresponds to old pair (0,1), rate=(.4-.8)/2.
        self.assertAlmostEqual(float(events['event_geometry'][1, 18]), -.2, places=5)


class BranchTests(unittest.TestCase):
    def batch(self):
        return {'event_appearance': torch.randn(2, 6, 16),
                'event_geometry': torch.randn(2, 6, 32),
                'event_reliability': torch.full((2, 6, 5), .9),
                'event_change': torch.ones(2, 6),
                'event_mask': torch.ones(2, 6, dtype=torch.bool)}

    def test_zero_initialization_and_nonzero_gate_gradient(self):
        for variant in ('B', 'C', 'D'):
            branch = RelationEventBranch(variant, appearance_dim=16, dim=16, output_dim=32, memory_slots=2)
            scene = torch.randn(2, 3, 32)
            result = branch(scene, self.batch())
            torch.testing.assert_close(result, scene, rtol=0, atol=0)
            result.square().mean().backward()
            self.assertTrue(bool(torch.isfinite(branch.scale.grad)))
            self.assertNotEqual(float(branch.scale.grad), 0.)

    def test_empty_objects_remain_finite_and_do_not_change_base(self):
        for variant in ('B', 'C', 'D'):
            branch = RelationEventBranch(variant, appearance_dim=16, dim=16, output_dim=32)
            branch.scale.data.fill_(.3)
            batch = self.batch(); batch['event_mask'].fill_(False)
            scene = torch.randn(2, 3, 32)
            torch.testing.assert_close(branch(scene, batch), scene, rtol=0, atol=0)

    def test_zero_reliability_disables_c_and_d(self):
        for variant in ('C', 'D'):
            branch = RelationEventBranch(variant, appearance_dim=16, dim=16, output_dim=32)
            branch.scale.data.fill_(.3)
            batch = self.batch(); batch['event_reliability'].zero_()
            scene = torch.randn(2, 3, 32)
            torch.testing.assert_close(branch(scene, batch), scene, rtol=0, atol=0)

    def test_padding_compaction_does_not_change_branch(self):
        for variant in ('B', 'C', 'D'):
            branch = RelationEventBranch(variant, appearance_dim=16, dim=16,
                                         output_dim=32, memory_slots=2).eval()
            branch.scale.data.fill_(.3)
            batch = self.batch()
            batch['event_mask'][:, 3:] = False
            compact = {key: value[:, :3].clone() for key, value in batch.items()}
            scene = torch.randn(2, 3, 32)
            torch.testing.assert_close(branch(scene, batch), branch(scene, compact),
                                       rtol=1e-5, atol=1e-6)

    def test_order_transform_and_caption_filter(self):
        batch = self.batch()
        changed = reverse_event_dynamics(batch)
        torch.testing.assert_close(changed['event_geometry'][..., 18:27], -batch['event_geometry'][..., 18:27])
        self.assertFalse(eligible_order_caption('A person is playing guitar.'))
        self.assertTrue(eligible_order_caption('A man enters the room then sits down.'))
        self.assertFalse(eligible_order_caption('A cat and a dog eat out of the same bowl.'))
        self.assertFalse(eligible_order_caption('A woman sings then a man wishes her happy birthday.'))
        self.assertTrue(eligible_order_caption('A police officer is approaching a couple.'))
        self.assertFalse(eligible_order_caption('A wrestler is getting ready to walk into a ring.'))
        good = torch.tensor([-.5], requires_grad=True)
        bad = torch.tensor([-1.])
        self.assertEqual(float(order_ranking_loss(good, bad, torch.ones(1))), 0.)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
