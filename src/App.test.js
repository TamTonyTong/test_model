import { render, screen } from '@testing-library/react';
import React from 'react';

jest.mock('react-router-dom', () => ({
  BrowserRouter: ({ children }) => <>{children}</>,
  Routes: ({ children }) => <>{children}</>,
  Route: () => null,
  NavLink: ({ children, to, className }) => {
    const computedClassName = typeof className === 'function'
      ? className({ isActive: false })
      : className;

    return (
      <a href={to} className={computedClassName}>
        {children}
      </a>
    );
  }
}), { virtual: true });

jest.mock('./Map', () => () => <div>Map Mock</div>);
jest.mock('./Predictor', () => () => <div>Predictor Mock</div>);
jest.mock('./About', () => () => <div>About Mock</div>);
jest.mock('./Contact', () => () => <div>Contact Mock</div>);
jest.mock('./Visualize', () => () => <div>Visualize Mock</div>);

import App from './App';

test('renders primary navigation links', () => {
  render(<App />);
  expect(screen.getByRole('link', { name: /predictor/i })).toBeInTheDocument();
  expect(screen.getByRole('link', { name: /map explorer/i })).toBeInTheDocument();
  expect(screen.getByRole('link', { name: /visualize/i })).toBeInTheDocument();
  expect(screen.getByRole('link', { name: /about/i })).toBeInTheDocument();
  expect(screen.getByRole('link', { name: /contact/i })).toBeInTheDocument();
});
