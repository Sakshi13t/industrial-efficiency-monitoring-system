import React, { createContext, useContext, useState, useEffect } from 'react';

const MonitoringContext = createContext();

import { API_BASE_URL } from '../lib/api';

export function MonitoringProvider({ children }) {
  const [activeSessions, setActiveSessionsState] = useState({});
  const [isValidating, setIsValidating] = useState(true);

  // On mount, validate sessions with backend
  useEffect(() => {
    const validateSessions = async () => {
      try {
        const saved = localStorage.getItem('activeSessions');
        if (!saved) {
          setIsValidating(false);
          return;
        }

        const savedSessions = JSON.parse(saved);
        const validatedSessions = {};

        // Check each saved session against the backend
        for (const [cameraId, sessionId] of Object.entries(savedSessions)) {
          try {
            const response = await fetch(`${API_BASE_URL}/monitor/metrics/${sessionId}`);
            if (response.ok) {
              const data = await response.json();
              // Only keep sessions that are still running on backend
              if (data.status === 'running') {
                validatedSessions[cameraId] = sessionId;
              }
            }
          } catch (error) {
            console.log(`Session ${sessionId} is no longer valid, removing...`);
          }
        }

        setActiveSessionsState(validatedSessions);
        localStorage.setItem('activeSessions', JSON.stringify(validatedSessions));
      } catch (error) {
        console.error('Error validating sessions:', error);
        // Clear invalid localStorage data
        localStorage.removeItem('activeSessions');
        setActiveSessionsState({});
      } finally {
        setIsValidating(false);
      }
    };

    validateSessions();
  }, []);

  // Persist active sessions to localStorage whenever they change
  useEffect(() => {
    if (!isValidating) {
      try {
        localStorage.setItem('activeSessions', JSON.stringify(activeSessions));
      } catch (error) {
        console.error('Error saving sessions to localStorage:', error);
      }
    }
  }, [activeSessions, isValidating]);

  // Custom setter that updates both state and localStorage
  const setActiveSessions = (updater) => {
    setActiveSessionsState(prev => {
      const newState = typeof updater === 'function' ? updater(prev) : updater;
      return newState;
    });
  };

  // Function to clear all sessions (useful for logout or reset)
  const clearAllSessions = () => {
    setActiveSessions({});
    localStorage.removeItem('activeSessions');
  };

  // Function to check if any sessions are active
  const hasActiveSessions = () => {
    return Object.keys(activeSessions).length > 0;
  };

  // Function to get session by camera ID
  const getSessionByCamera = (cameraId) => {
    return activeSessions[cameraId] || null;
  };

  // Function to manually validate a single session
  const validateSession = async (cameraId, sessionId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/monitor/metrics/${sessionId}`);
      if (!response.ok) {
        // Session is invalid, remove it
        setActiveSessions(prev => {
          const updated = { ...prev };
          delete updated[cameraId];
          return updated;
        });
        return false;
      }
      return true;
    } catch (error) {
      // Session is invalid, remove it
      setActiveSessions(prev => {
        const updated = { ...prev };
        delete updated[cameraId];
        return updated;
      });
      return false;
    }
  };

  return (
    <MonitoringContext.Provider value={{ 
      activeSessions, 
      setActiveSessions,
      clearAllSessions,
      hasActiveSessions,
      getSessionByCamera,
      validateSession,
      isValidating
    }}>
      {children}
    </MonitoringContext.Provider>
  );
}

export function useMonitoring() {
  const context = useContext(MonitoringContext);
  if (!context) {
    throw new Error('useMonitoring must be used within MonitoringProvider');
  }
  return context;
}

// import React, { createContext, useContext, useState, useEffect } from 'react';

// const MonitoringContext = createContext();

// export function MonitoringProvider({ children }) {
//   // Load active sessions from localStorage on mount (persistence across page reloads)
//   const [activeSessions, setActiveSessionsState] = useState(() => {
//     try {
//       const saved = localStorage.getItem('activeSessions');
//       return saved ? JSON.parse(saved) : {};
//     } catch (error) {
//       console.error('Error loading sessions from localStorage:', error);
//       return {};
//     }
//   });

//   // Persist active sessions to localStorage whenever they change
//   useEffect(() => {
//     try {
//       localStorage.setItem('activeSessions', JSON.stringify(activeSessions));
//     } catch (error) {
//       console.error('Error saving sessions to localStorage:', error);
//     }
//   }, [activeSessions]);

//   // Custom setter that updates both state and localStorage
//   const setActiveSessions = (updater) => {
//     setActiveSessionsState(prev => {
//       const newState = typeof updater === 'function' ? updater(prev) : updater;
//       return newState;
//     });
//   };

//   // Function to clear all sessions (useful for logout or reset)
//   const clearAllSessions = () => {
//     setActiveSessions({});
//     localStorage.removeItem('activeSessions');
//   };

//   // Function to check if any sessions are active
//   const hasActiveSessions = () => {
//     return Object.keys(activeSessions).length > 0;
//   };

//   // Function to get session by camera ID
//   const getSessionByCamera = (cameraId) => {
//     return activeSessions[cameraId] || null;
//   };

//   return (
//     <MonitoringContext.Provider value={{ 
//       activeSessions, 
//       setActiveSessions,
//       clearAllSessions,
//       hasActiveSessions,
//       getSessionByCamera
//     }}>
//       {children}
//     </MonitoringContext.Provider>
//   );
// }

// export function useMonitoring() {
//   const context = useContext(MonitoringContext);
//   if (!context) {
//     throw new Error('useMonitoring must be used within MonitoringProvider');
//   }
//   return context;
// }

