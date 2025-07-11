import React from 'react';
import { motion } from 'framer-motion';
import { Waves, Loader2 } from 'lucide-react';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  text?: string;
  variant?: 'default' | 'water' | 'minimal';
  className?: string;
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'md',
  text = 'Loading...',
  variant = 'default',
  className = ''
}) => {
  const sizeClasses = {
    sm: 'w-8 h-8',
    md: 'w-12 h-12',
    lg: 'w-16 h-16'
  };

  const textSizeClasses = {
    sm: 'text-sm',
    md: 'text-base',
    lg: 'text-lg'
  };

  if (variant === 'minimal') {
    return (
      <div className={`flex items-center justify-center ${className}`}>
        <Loader2 className={`${sizeClasses[size]} animate-spin text-primary`} />
      </div>
    );
  }

  if (variant === 'water') {
    return (
      <div className={`flex flex-col items-center justify-center space-y-4 p-8 ${className}`}>
        <motion.div
          animate={{
            scale: [1, 1.1, 1],
            rotate: [0, 180, 360]
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: "easeInOut"
          }}
          className="relative"
        >
          <Waves className={`${sizeClasses[size]} text-blue-500`} />
          <motion.div
            animate={{
              scale: [0.8, 1.2, 0.8],
              opacity: [0.3, 0.8, 0.3]
            }}
            transition={{
              duration: 1.5,
              repeat: Infinity,
              ease: "easeInOut"
            }}
            className="absolute inset-0 rounded-full bg-blue-200 dark:bg-blue-800"
          />
        </motion.div>
        {text && (
          <motion.p
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 1.5, repeat: Infinity }}
            className={`${textSizeClasses[size]} text-muted-foreground font-medium`}
          >
            {text}
          </motion.p>
        )}
      </div>
    );
  }

  return (
    <div className={`flex flex-col items-center justify-center p-8 ${className}`}>
      <div className="relative">
        <motion.div
          className={`${sizeClasses[size]} border-4 border-primary/20 border-t-primary rounded-full`}
          animate={{ rotate: 360 }}
          transition={{
            duration: 1.2,
            repeat: Infinity,
            ease: "linear"
          }}
        />
        <div className="absolute inset-0 flex items-center justify-center">
          <motion.div
            className="w-2/3 h-2/3 rounded-full"
            animate={{
              boxShadow: [
                '0 0 0 0 rgba(59, 130, 246, 0)',
                '0 0 0 10px rgba(59, 130, 246, 0.1)',
                '0 0 0 0 rgba(59, 130, 246, 0)'
              ]
            }}
            transition={{
              repeat: Infinity,
              duration: 1.5
            }}
          />
        </div>
      </div>
      {text && (
        <motion.p
          className={`mt-4 ${textSizeClasses[size]} text-muted-foreground font-medium`}
          animate={{ opacity: [0.5, 1, 0.5] }}
          transition={{
            repeat: Infinity,
            duration: 1.5
          }}
        >
          {text}
        </motion.p>
      )}
    </div>
  )
}

export default LoadingSpinner 