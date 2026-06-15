import React, { useEffect, useState } from 'react';
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom';
import Cempack from "../assets/Cempack.svg"
import Cube from "../assets/Cube.svg"
import Optima from "../assets/Optima.svg"
import {
    LayoutDashboard,
    Settings,
    Activity,
    FileBarChart,
    HelpCircle,
    LogOut,
    Box,
    Video
} from 'lucide-react';

export default function Layout() {
    const navigate = useNavigate();
    const location = useLocation();

    const handleLogout = () => {
        localStorage.removeItem('isAuthenticated');
        navigate('/login');
    };
    
    return (
        <div className="flex bg-primary min-h-screen">
            {/* Sidebar */}
            <aside className="sidebar flex justify-center items-center ">
                <div className="logo-area ">
                    {/* <div className="logo-icon flex items-center justify-center">
                        <Box size={20} color="white" />
                    </div> */}
                    <div className='w-80 '>
                        {/* <h2 className="font-bold text-lg tracking-tight">CemPack Optima</h2> */}

                        {/* <svg className="relative bottom-4 right-4" width="540" height="120" viewBox="0 0 540 120" xmlns="http://www.w3.org/2000/svg "> */}
{/* <!-- Logo Mark --> */}
{/* <g transform="translate(40,36)">
<circle cx="25" cy="25" r="14"

            fill="none"

            stroke="#57C785"

            stroke-width="4"/>
<circle cx="18" cy="18" r="4"

            fill="#3B82F6"/>
</g> */}
 
  {/* <!-- Brand --> */}
{/* <text x="96" y="64"

        font-family="Poppins, Inter, Arial, sans-serif"

        font-size="40"

        font-weight="600"

        fill="white"

        letter-spacing="-0.6">

    Cempack */}
{/* </text> */}
 
  {/* <text x="96" y="96"

        font-family="Poppins, Inter, Arial, sans-serif"

        font-size="28"

        font-weight="400"

        fill="#3B82F6"

        letter-spacing="-0.2">

    Optima
</text> */}
{/* </svg> */}
   <img className='mx-auto mb-5 w-15'  src={Cube} alt='error'/>
              <img className="ml-17   text-amber-200" src={Cempack} alt='error'/>
               <img  className='ml-25 w-25 h-10 mb-7 ' src={Optima} alt='error'/>
           
                        <p className="text-xs text-secondary">Efficiency AI</p>
                    </div>
                </div>

                <nav className="flex-1 space-y-1">
                    <NavLink to="/" className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`}>
                        <LayoutDashboard size={20} /> Dashboard
                    </NavLink>

                      <NavLink to="/cameras" className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`}>
                        <Video size={20} /> Camera Settings
                    </NavLink>

                      <NavLink to="/packers" className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`}>
                        <Settings size={20} /> Packer Master
                    </NavLink>

                    <NavLink to="/monitoring" className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`}>
                        <Activity size={20} /> Live Monitoring
                    </NavLink>

                    <NavLink to="/reports" className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`}>
                        <FileBarChart size={20} /> Analysis History
                    </NavLink>
                    <NavLink to="/support" className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`}>
                        <HelpCircle size={20} /> Feedback & Support
                    </NavLink>
                </nav>

                <div className="mt-auto flex items-center justify-center bg-gray-600 hover:bg-gray-500 w-20 h-8 rounded-xl  ">
                    <div>
                         <LogOut size={16} className='relative left-1 top-0.4'/>
                    </div>
                    <button onClick={handleLogout} className=' w-full font-light  ' >
                        Logout
                    </button>

                </div>
            </aside>

            {/* Main Content */}
            <main className="main-content w-full">
                <Outlet />
            </main>
        </div>
    );
}
