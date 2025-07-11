import React, { useState, useRef, useEffect, ReactNode } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Loader2, Eye, Zap } from 'lucide-react';

interface LazyWrapperProps {
  children: ReactNode;
  fallback?: ReactNode;
  threshold?: number;
  rootMargin?: string;
  delay?: number;
  animation?: 'fade' | 'slide' | 'scale' | 'rotate' | 'bounce' | 'flip';
  direction?: 'up' | 'down' | 'left' | 'right';
  className?: string;
  once?: boolean;
  onVisible?: () => void;
  onHidden?: () => void;
  showProgress?: boolean;
  enableHover?: boolean;
}

const LazyWrapper: React.FC<LazyWrapperProps> = ({
  children,
  fallback,
  threshold = 0.1,
  rootMargin = '50px',
  delay = 0,
  animation = 'fade',
  direction = 'up',
  className = '',
  once = true,
  onVisible,
  onHidden,
  showProgress = false,
  enableHover = true,
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const [progress, setProgress] = useState(0);
  const [isHovered, setIsHovered] = useState(false);
  const elementRef = useRef<HTMLDivElement>(null);
  const observerRef = useRef<IntersectionObserver | null>(null);

  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;

    observerRef.current = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setIsVisible(true);
            onVisible?.();

            // Simulate loading progress
            if (showProgress) {
              let currentProgress = 0;
              const progressInterval = setInterval(() => {
                currentProgress += Math.random() * 30;
                if (currentProgress >= 100) {
                  currentProgress = 100;
                  clearInterval(progressInterval);
                  setTimeout(() => setIsLoaded(true), delay);
                } else {
                  setProgress(currentProgress);
                }
              }, 100);
            } else {
              setTimeout(() => setIsLoaded(true), delay);
            }

            if (once) {
              observerRef.current?.unobserve(element);
            }
          } else if (!once) {
            setIsVisible(false);
            setIsLoaded(false);
            setProgress(0);
            onHidden?.();
          }
        });
      },
      {
        threshold,
        rootMargin,
      }
    );

    observerRef.current.observe(element);

    return () => {
      observerRef.current?.disconnect();
    };
  }, [threshold, rootMargin, delay, once, onVisible, onHidden, showProgress]);

  const getAnimationVariants = () => {
    const baseVariants = {
      hidden: { opacity: 0 },
      visible: { opacity: 1 },
    };

    switch (animation) {
      case 'slide':
        return {
          hidden: {
            opacity: 0,
            x: direction === 'left' ? -100 : direction === 'right' ? 100 : 0,
            y: direction === 'up' ? 100 : direction === 'down' ? -100 : 0,
          },
          visible: {
            opacity: 1,
            x: 0,
            y: 0,
          },
        };
      case 'scale':
        return {
          hidden: { opacity: 0, scale: 0.8 },
          visible: { opacity: 1, scale: 1 },
        };
      case 'rotate':
        return {
          hidden: { opacity: 0, rotate: -180, scale: 0.8 },
          visible: { opacity: 1, rotate: 0, scale: 1 },
        };
      case 'bounce':
        return {
          hidden: { opacity: 0, y: 100, scale: 0.3 },
          visible: {
            opacity: 1,
            y: 0,
            scale: 1,
            transition: {
              type: 'spring',
              damping: 10,
              stiffness: 100,
            },
          },
        };
      case 'flip':
        return {
          hidden: { opacity: 0, rotateY: -90, scale: 0.8 },
          visible: { opacity: 1, rotateY: 0, scale: 1 },
        };
      default:
        return baseVariants;
    }
  };

  const hoverVariants = enableHover ? {
    hover: {
      scale: 1.02,
      y: -4,
      transition: { duration: 0.2 },
    },
  } : {};

  const DefaultFallback = () => (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="flex flex-col items-center justify-center p-8 space-y-4"
    >
      <div className="relative">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
          className="w-8 h-8 border-2 border-primary-500 border-t-transparent rounded-full"
        />
        <motion.div
          animate={{ scale: [1, 1.2, 1] }}
          transition={{ duration: 1, repeat: Infinity }}
          className="absolute inset-0 w-8 h-8 border border-primary-300 rounded-full"
        />
      </div>

      {showProgress && (
        <div className="w-full max-w-xs">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">Loading...</span>
            <span className="text-sm font-medium text-primary-600 dark:text-primary-400">
              {Math.round(progress)}%
            </span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
            <motion.div
              className="bg-gradient-to-r from-primary-500 to-accent-500 h-2 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
        </div>
      )}

      <motion.p
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="text-sm text-gray-500 dark:text-gray-400 text-center"
      >
        Preparing amazing content...
      </motion.p>
    </motion.div>
  );

  return (
    <div
      ref={elementRef}
      className={`lazy-wrapper ${className}`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <AnimatePresence mode="wait">
        {!isLoaded ? (
          <motion.div
            key="fallback"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="min-h-[200px] flex items-center justify-center"
          >
            {fallback || <DefaultFallback />}
          </motion.div>
        ) : (
          <motion.div
            key="content"
            variants={getAnimationVariants()}
            initial="hidden"
            animate="visible"
            exit="hidden"
            whileHover={enableHover ? "hover" : undefined}
            transition={{
              duration: 0.6,
              ease: [0.25, 0.46, 0.45, 0.94],
              delay: delay / 1000,
            }}
            className="w-full"
          >
            {children}
          </motion.div>
        )}
      </AnimatePresence>


    </div>
  );
};

export default LazyWrapper;
