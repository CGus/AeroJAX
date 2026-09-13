import copy, os, time, unittest
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import jax,numpy as np
from PyQt6.QtWidgets import QApplication,QGridLayout
from solver import GridParams,FlowParams,FlowConstraints,GeometryParams,SimulationParams,BaselineSolver
from viewer.simulation_controller import SimulationController,MetricsWorker,SharedData
class AuditTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.app=QApplication.instance() or QApplication([])
  cls.s=BaselineSolver(GridParams(64,32,20.,7.5),FlowParams(U_inf=1.,nu=.001,Re=1000.,constraints=FlowConstraints(True,True,False)),GeometryParams(5.,3.75,.5),SimulationParams(adaptive_dt=False,obstacle_type='cylinder'),dt=.001)
 def solver(self):
  s=copy.copy(self.s);s.flow=copy.copy(s.flow);s.sim_params=copy.copy(s.sim_params)
  s._jit_cache={};s._batch_jit=None;s.iteration=0;s.simulated_time=0.;return s
 def test_batch_equivalence(self):
  a,b=self.solver(),self.solver();a.advance_steps(8)
  for _ in range(8):b.advance_steps(1)
  for k in ['u','v','current_pressure','u_prev','v_prev']:np.testing.assert_allclose(getattr(a,k),getattr(b,k),rtol=1e-5,atol=1e-6)
  self.assertAlmostEqual(a.simulated_time,.008)
 def test_dt_change(self):
  s=self.solver();s.advance_steps(1);s.dt=.0005
  expected=s._step(s.u,s.v,s.mask,s.dt,s.iteration);s.advance_steps(1)
  np.testing.assert_allclose(s.u,expected[0],rtol=1e-5,atol=1e-6);self.assertAlmostEqual(s.simulated_time,.0015)
 def test_flow_change(self):
  s=self.solver();s.advance_steps(1);old=s._step_jit;s.flow.U_inf=2.;s.flow.nu=.002;s.advance_steps(1)
  self.assertIsNot(s._step_jit,old);self.assertAlmostEqual(float(s.u[0,16]),2.)
 def test_adaptive(self):
  s=self.solver();s.sim_params.adaptive_dt=True;s.flow.U_inf=10.;s.advance_steps(4)
  self.assertLessEqual(s.dt,.035*min(s.grid.dx,s.grid.dy)/15+1e-9);self.assertGreater(s.simulated_time,0.);self.assertTrue(np.isfinite(s.u).all())
 def test_display_read_only(self):
  s=self.solver();s.visualization_fields();self.assertEqual(s.iteration,0)
 def test_shared_cleanup(self):
  x=SharedData((2,2));x.cleanup();x.cleanup()
 def test_lifecycle(self):
  s=self.solver();s.advance_steps(8);jax.block_until_ready(s.u)
  c=SimulationController(s);c.start_simulation({});w=c.simulation_worker;self.assertEqual(w.shared_buffers,{})
  time.sleep(.1);c.pause_simulation();n=s.iteration;time.sleep(.05);self.assertEqual(s.iteration,n)
  c.resume_simulation();time.sleep(.05);c.stop_simulation();self.assertFalse(w.thread.is_alive());self.assertIsNone(c.metrics_worker);c.stop_simulation();c.full_reset()
 def test_metrics_snapshot(self):
  s=self.solver();m=MetricsWorker(s);m.enqueue_data(s.u,s.v,s.current_pressure,s.mask,0);snap=m.data_queue.get_nowait()[-1]
  previous=snap.u_prev;s.u_prev=s.u*3;s.flow.U_inf=7;self.assertIs(snap.u_prev,previous);self.assertNotEqual(snap.flow.U_inf,7)
  m.start();t=m.thread;m.start();self.assertIs(m.thread,t);m.stop();m.stop()
 def test_reynolds_layout(self):
  from viewer.ui_components.control_panel import ControlPanel
  cp=ControlPanel();layouts=cp.re_group.findChildren(QGridLayout);self.assertTrue(layouts)
  for layout in layouts:
   occupied=set()
   for i in range(layout.count()):
    row,col,rs,cs=layout.getItemPosition(i)
    for rr in range(row,row+rs):
     for cc in range(col,col+cs):self.assertNotIn((rr,cc),occupied);occupied.add((rr,cc))
  cp.re_auto_combo.setCurrentText('ν');self.assertTrue(cp.nu_input.isReadOnly());self.assertFalse(cp.u_input.isReadOnly());cp.deleteLater()
 def test_failure_stops_worker_and_identifies_source(self):
  from viewer.simulation_controller import SimulationWorker
  s=self.solver()
  def fail(count):raise RuntimeError('expected regression-test failure')
  s.advance_steps=fail
  w=SimulationWorker(s);events=[]
  w.failed.connect(lambda source,message:events.append((source,message)))
  w.start();w.thread.join(timeout=3);self.app.processEvents()
  self.assertFalse(w.running);self.assertFalse(w.thread.is_alive())
  self.assertEqual(events,[(w,'expected regression-test failure')]);w.stop_simulation()
if __name__=='__main__':unittest.main()
