import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { MonitoringProvider } from './contexts/MonitoringContext';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Monitoring from './pages/Monitoring';
import PackerMaster from './pages/PackerMaster';
import CameraMaster from './pages/CameraMaster';
import Reports from './pages/Reports';
import Support from './pages/Support';
import Login from './pages/Login';

function PrivateRoute({ children }) {
  const isAuthenticated = localStorage.getItem('isAuthenticated') === 'true';
  return isAuthenticated ? children : <Navigate to="/login" />;
}

function App() {
  return (
    <BrowserRouter>
    <MonitoringProvider>
      <Routes>
        <Route path="/login" element={<Login />} />

        <Route element={<PrivateRoute><Layout /></PrivateRoute>}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/monitoring" element={<Monitoring />} />
          <Route path="/packers" element={<PackerMaster />} />
          <Route path="/cameras" element={<CameraMaster />} />
          <Route path="/reports" element={<Reports />} />
          <Route path="/support" element={<Support />} />
        </Route>
      </Routes>
    </MonitoringProvider>
    </BrowserRouter>
  );
}

export default App;
