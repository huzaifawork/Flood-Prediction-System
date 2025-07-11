import React, { ReactNode, useState } from 'react';
import { motion } from 'framer-motion';
import { Sparkles, Star } from 'lucide-react';

interface EnhancedCardProps {
  children: ReactNode;
  variant?: 'default' | 'glass' | 'gradient';
  hover?: 'lift' | 'scale' | 'none';
  animation?: 'breathe' | 'none';
  interactive?: boolean;
  className?: string;
  onClick?: () => void;
  showParticles?: boolean;
  rippleEffect?: boolean;
}

const EnhancedCard: React.FC<EnhancedCardProps> = ({
  children,
  variant = 'default',
  hover = 'lift',
  animation = 'none',
  interactive = true,
  className = '',
  onClick,
  showParticles = false,
  rippleEffect = false,
}) => {
  const [isHovered, setIsHovered] = useState(false);
  const [ripples, setRipples] = useState<Array<{ id: number; x: number; y: number }>>([]);

  const getVariantClasses = () => {
    const baseClasses = 'relative overflow-hidden transition-all duration-300 ease-out p-6 rounded-2xl';

    switch (variant) {
      case 'glass':
        return `${baseClasses} card-glass backdrop-blur-xl border border-white/20 dark:border-white/10`;
      case 'gradient':
        return `${baseClasses} card-gradient`;
      default:
        return `${baseClasses} card`;
    }
  };

  const getHoverVariants = () => {
    if (!interactive || hover === 'none') return {};

    switch (hover) {
      case 'lift':
        return {
          hover: {
            y: -8,
            scale: 1.02,
            transition: { duration: 0.3, ease: 'easeOut' },
          },
        };
      case 'scale':
        return {
          hover: {
            scale: 1.05,
            transition: { duration: 0.3, ease: 'easeOut' },
          },
        };
      default:
        return {};
    }
  };

  const getAnimationClasses = () => {
    switch (animation) {
      case 'breathe':
        return ''; // Breathing animation removed per user request
      default:
        return '';
    }
  };

  const handleMouseEnter = () => {
    setIsHovered(true);
  };

  const handleMouseLeave = () => {
    setIsHovered(false);
  };

  const handleClick = (e: React.MouseEvent) => {
    if (rippleEffect) {
      const rect = e.currentTarget.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const newRipple = { id: Date.now(), x, y };
      setRipples(prev => [...prev, newRipple]);

      setTimeout(() => {
        setRipples(prev => prev.filter(ripple => ripple.id !== newRipple.id));
      }, 600);
    }

    onClick?.();
  };

  const Particles = () => {
    if (!showParticles || !isHovered) return null;

    return (
      <div className="absolute inset-0 pointer-events-none">
        {Array.from({ length: 3 }).map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-primary-400 rounded-full"
            initial={{
              x: Math.random() * 100 + '%',
              y: Math.random() * 100 + '%',
              opacity: 0,
              scale: 0,
            }}
            animate={{
              y: [null, '-20px', '-40px'],
              opacity: [0, 1, 0],
              scale: [0, 1, 0],
            }}
            transition={{
              duration: 2,
              delay: i * 0.1,
              repeat: Infinity,
              ease: 'easeOut',
            }}
          />
        ))}
      </div>
    );
  };

  const Ripples = () => (
    <>
      {ripples.map((ripple) => (
        <motion.div
          key={ripple.id}
          className="absolute bg-white/30 rounded-full pointer-events-none"
          style={{
            left: ripple.x - 25,
            top: ripple.y - 25,
          }}
          initial={{ width: 0, height: 0, opacity: 0.5 }}
          animate={{ width: 100, height: 100, opacity: 0 }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
        />
      ))}
    </>
  );

  return (
    <motion.div
      className={`
        ${getVariantClasses()}
        ${getAnimationClasses()}
        ${interactive ? 'cursor-pointer' : ''}
        ${className}
      `}
      variants={getHoverVariants()}
      whileHover="hover"
      whileTap={interactive ? { scale: 0.98 } : undefined}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onClick={handleClick}
    >
      {/* Background gradient overlay */}
      <motion.div
        className="absolute inset-0 bg-gradient-to-br from-primary-500/5 via-transparent to-purple-500/5 opacity-0 transition-opacity duration-500"
        animate={{ opacity: isHovered ? 1 : 0 }}
      />

      {/* Content */}
      <div className="relative z-10">
        {children}
      </div>

      {/* Interactive elements */}
      <Particles />
      <Ripples />

      {/* Corner decorations */}
      {isHovered && (
        <>
          <motion.div
            className="absolute top-2 right-2 text-primary-400"
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.1 }}
          >
            <Sparkles className="w-4 h-4" />
          </motion.div>
          <motion.div
            className="absolute bottom-2 left-2 text-purple-400"
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 }}
          >
            <Star className="w-3 h-3" />
          </motion.div>
        </>
      )}
    </motion.div>
  );
};

export default EnhancedCard;
