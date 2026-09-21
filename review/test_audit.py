"""Synthetic tests for the independent paper audit; no live services."""
import unittest
from audit import integrate, system_energy, system_saving, operational_carbon, operational_water, replay, routed_footprint


class AuditChecks(unittest.TestCase):
    def test_clipped_ramp(self):
        # P(t)=10t+10 over t in [0,2]: exactly 40 J.
        self.assertAlmostEqual(integrate([(-1,0),(1,20),(3,40)],2),40/3600)

    def test_missing_bracket(self):
        with self.assertRaises(ValueError):
            integrate([(0,10),(1,10)],2)

    def test_supply_conversion_applies_at_zero_host(self):
        self.assertAlmostEqual(system_energy(.18,3,0,.9),.20)
        self.assertAlmostEqual(system_energy(.18,3,60,.9),.23/.9)

    def test_same_volume_growth_for_all_components(self):
        self.assertAlmostEqual(system_saving(1,.25,.8,.25),.5)
        self.assertAlmostEqual(system_saving(2,.5,.8,.25),.5)

    def test_no_routing_still_charges_new_usage(self):
        self.assertAlmostEqual(system_saving(1,.25,0,.2),-.2)

    def test_router_overhead(self):
        self.assertAlmostEqual(system_saving(1,.25,1,0,.1),.65)

    def test_water_units(self):
        self.assertAlmostEqual(operational_water(2,2,.4,1),2.4)

    def test_carbon_units(self):
        self.assertAlmostEqual(operational_carbon(2,500),1)

    def test_growth_break_even(self):
        self.assertAlmostEqual(system_saving(1,.25,.8,1.5),0)

    def test_failed_local_attempt_also_pays_cloud(self):
        self.assertAlmostEqual(routed_footprint(1,.25,.8,.1),.48)

    def test_full_archived_dataset_replay(self):
        report = replay()
        self.assertEqual(report['integrated_requests'],38)
        self.assertEqual(report['raw_repetitions'],68)
        self.assertEqual(len(report['ladder_rows']),12)
        self.assertEqual(sum(not r['raw_available'] for r in report['ladder_rows']),9)
        self.assertLess(report['max_energy_rounding_difference_wh'],.00000051)


if __name__ == '__main__':
    unittest.main()
