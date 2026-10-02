import random
import unittest

from audio_transcript.domain.models import SegmentationSettings, VoiceActivity
from audio_transcript.domain.segmentation import plan_chunks

FRAME = 0.032
TOLERANCE = 1e-9


def activity(pattern, frame=FRAME, threshold=0.5):
    """Build probabilities from (seconds, probability) runs."""
    values = []
    for seconds, probability in pattern:
        values.extend([probability] * round(seconds / frame))
    return VoiceActivity(frame, tuple(values), threshold, "test")


def duration_of(voice):
    return len(voice.probabilities) * voice.frame_seconds


def assert_invariants(test, regions, total, settings):
    previous_end = 0.0
    for region in regions:
        test.assertGreaterEqual(region.start, previous_end - TOLERANCE)
        test.assertGreater(region.end, region.start)
        test.assertLessEqual(region.end - region.start, settings.max_chunk_seconds + TOLERANCE)
        test.assertGreaterEqual(region.start, 0.0)
        test.assertLessEqual(region.end, total + TOLERANCE)
        previous_end = region.end


class PlanChunksTests(unittest.TestCase):
    def setUp(self):
        self.settings = SegmentationSettings()

    def test_all_silence_yields_no_chunks(self):
        voice = activity([(30, 0.01)])
        self.assertEqual(plan_chunks(voice, duration_of(voice), self.settings), [])
        self.assertEqual(plan_chunks(activity([]), 10.0, self.settings), [])

    def test_single_phrase_is_padded_and_clamped(self):
        voice = activity([(2, 0.0), (3, 0.9), (2, 0.0)])
        regions = plan_chunks(voice, duration_of(voice), self.settings)
        self.assertEqual(len(regions), 1)
        self.assertAlmostEqual(regions[0].start, 2.0 - 0.2, delta=FRAME)
        self.assertAlmostEqual(regions[0].end, 5.0 + 0.2, delta=FRAME)
        edge = activity([(3, 0.9), (0.5, 0.0)])
        first = plan_chunks(edge, duration_of(edge), self.settings)[0]
        self.assertEqual(first.start, 0.0)

    def test_short_speech_bursts_are_dropped(self):
        voice = activity([(2, 0.0), (0.1, 0.9), (2, 0.0)])
        self.assertEqual(plan_chunks(voice, duration_of(voice), self.settings), [])

    def test_hysteresis_keeps_speech_with_intermediate_probability(self):
        # 0.4 is below the onset (0.5) but above the offset (0.35): still speech.
        voice = activity([(1, 0.0), (1, 0.9), (1, 0.4), (1, 0.9), (1, 0.0)])
        regions = plan_chunks(
            voice, duration_of(voice), SegmentationSettings(max_merge_gap_seconds=0)
        )
        self.assertEqual(len(regions), 1)

    def test_brief_dip_does_not_split_speech(self):
        voice = activity([(1, 0.0), (2, 0.9), (0.064, 0.0), (2, 0.9), (1, 0.0)])
        regions = plan_chunks(
            voice, duration_of(voice), SegmentationSettings(max_merge_gap_seconds=0)
        )
        self.assertEqual(len(regions), 1)

    def test_nearby_phrases_merge_and_far_phrases_stay_separate(self):
        voice = activity([(1, 0), (2, 0.9), (0.6, 0), (2, 0.9), (4, 0), (2, 0.9), (1, 0)])
        regions = plan_chunks(voice, duration_of(voice), self.settings)
        self.assertEqual(len(regions), 2)
        self.assertAlmostEqual(regions[0].end - regions[0].start, 5.0, delta=0.1)

    def test_merge_respects_max_chunk(self):
        voice = activity([(1, 0), (8, 0.9), (0.5, 0), (8, 0.9), (1, 0)])
        regions = plan_chunks(voice, duration_of(voice), self.settings)
        self.assertEqual(len(regions), 2)
        assert_invariants(self, regions, duration_of(voice), self.settings)

    def test_overlapping_padding_is_split_between_neighbours(self):
        settings = SegmentationSettings(
            max_chunk_seconds=4.5, max_merge_gap_seconds=1, speech_pad_seconds=0.5
        )
        voice = activity([(1, 0), (2, 0.9), (0.4, 0), (2, 0.9), (1, 0)])
        regions = plan_chunks(voice, duration_of(voice), settings)
        self.assertEqual(len(regions), 2)
        self.assertLessEqual(regions[0].end, regions[1].start + TOLERANCE)

    def test_long_run_is_cut_at_last_pause_inside_the_window(self):
        pattern = [(1, 0)] + [(5, 0.9), (0.16, 0.0)] * 4 + [(5, 0.9), (1, 0)]
        voice = activity(pattern)
        # Brief pauses (5 frames) are shorter than min_silence_seconds=0.4 so they stay inside runs.
        settings = SegmentationSettings(min_silence_seconds=0.4, max_merge_gap_seconds=0)
        regions = plan_chunks(voice, duration_of(voice), settings)
        assert_invariants(self, regions, duration_of(voice), settings)
        self.assertGreaterEqual(len(regions), 2)
        pause_starts = [1 + 5.16 * k - 0.16 for k in range(1, 5)]
        cut = regions[0].end
        # The first cut falls inside one of the brief pauses (plus padding at most).
        self.assertTrue(any(abs(cut - (start + 0.08)) < 0.35 for start in pause_starts), cut)

    def test_continuous_speech_is_hard_cut_below_the_maximum(self):
        voice = activity([(1, 0), (60, 0.9), (1, 0)])
        regions = plan_chunks(voice, duration_of(voice), self.settings)
        assert_invariants(self, regions, duration_of(voice), self.settings)
        self.assertGreaterEqual(len(regions), 5)
        covered = sum(region.end - region.start for region in regions)
        self.assertGreater(covered, 60.0)

    def test_hard_cut_prefers_lowest_probability_frame(self):
        values = [0.0] * 30 + [0.9] * 600
        values[30 + 380] = 0.45  # a single weak frame, never below the offset
        voice = VoiceActivity(FRAME, tuple(values), 0.5, "test")
        settings = SegmentationSettings(max_chunk_seconds=15, max_merge_gap_seconds=0)
        regions = plan_chunks(voice, duration_of(voice), settings)
        self.assertAlmostEqual(regions[0].end, (30 + 380) * FRAME, delta=0.35)

    def test_invariants_hold_for_random_activity_with_a_fixed_seed(self):
        rng = random.Random(20260502)
        for trial in range(300):
            settings = SegmentationSettings(
                max_chunk_seconds=rng.choice([2.0, 5.0, 15.0, 30.0]),
                min_speech_seconds=rng.choice([0.1, 0.25, 0.5]),
                min_silence_seconds=rng.choice([0.05, 0.1, 0.3]),
                speech_pad_seconds=rng.choice([0.0, 0.2, 0.6]),
                max_merge_gap_seconds=rng.choice([0.0, 1.0, 3.0]),
            )
            frames = rng.randint(1, 3000)
            values = []
            level = rng.random()
            while len(values) < frames:
                run = rng.randint(1, 120)
                level = rng.choice([0.0, 0.2, 0.4, 0.6, 0.95, rng.random()])
                values.extend([level] * run)
            values = values[:frames]
            voice = VoiceActivity(FRAME, tuple(values), rng.choice([0.4, 0.5, 0.6]), "test")
            total = frames * FRAME - rng.choice([0.0, 0.01])
            regions = plan_chunks(voice, total, settings)
            assert_invariants(self, regions, total, settings)
            self.assertEqual(regions, plan_chunks(voice, total, settings), trial)

    def test_planning_is_deterministic(self):
        voice = activity([(1, 0), (40, 0.9), (1, 0), (3, 0.8), (1, 0)])
        first = plan_chunks(voice, duration_of(voice), self.settings)
        self.assertEqual(first, plan_chunks(voice, duration_of(voice), self.settings))

    def test_invalid_settings_are_rejected(self):
        voice = activity([(1, 0.9)])
        with self.assertRaises(ValueError):
            plan_chunks(voice, 1.0, SegmentationSettings(max_chunk_seconds=0))
        with self.assertRaises(ValueError):
            plan_chunks(VoiceActivity(0.0, (0.9,), 0.5, "x"), 1.0, self.settings)


if __name__ == "__main__":
    unittest.main()
