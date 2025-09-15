# Contributing to Executive CRM Dashboard

Thank you for your interest in contributing to the Executive CRM Dashboard! This document provides guidelines for contributing to this project.

## 🤝 Code of Conduct

This project follows a Code of Conduct. By participating, you are expected to uphold this code.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Basic knowledge of Streamlit and data visualization

### Development Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/executive-crm-dashboard.git
cd executive-crm-dashboard

# Create virtual environment
python -m venv crm_env
source crm_env/bin/activate  # Windows: crm_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Run the application
streamlit run app_enhanced.py
```

## 🛠️ Development Guidelines

### Code Style
- Follow PEP 8 Python style guide
- Use type hints where applicable
- Write docstrings for all functions and classes
- Keep functions focused and single-purpose

### Naming Conventions
- Use descriptive variable and function names
- Use snake_case for functions and variables
- Use PascalCase for classes
- Use UPPER_CASE for constants

### Testing
- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Aim for at least 80% code coverage

## 📝 How to Contribute

### Reporting Issues
1. Check existing issues to avoid duplicates
2. Use the issue template
3. Provide detailed information:
   - Steps to reproduce
   - Expected behavior
   - Actual behavior
   - Screenshots if applicable
   - Environment details

### Suggesting Features
1. Check existing feature requests
2. Describe the feature and its benefits
3. Provide use cases and examples
4. Consider implementation complexity

### Submitting Code Changes

#### Pull Request Process
1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Make** your changes
4. **Test** your changes thoroughly
5. **Commit** your changes: `git commit -m 'Add amazing feature'`
6. **Push** to your branch: `git push origin feature/amazing-feature`
7. **Open** a Pull Request

#### Pull Request Guidelines
- Use the PR template
- Write clear, concise commit messages
- Keep changes focused and atomic
- Update documentation as needed
- Add tests for new functionality
- Ensure CI passes

## 🎯 Areas for Contribution

### High Priority
- [ ] Additional AI/ML models integration
- [ ] Mobile responsiveness improvements
- [ ] Performance optimizations
- [ ] Data export functionality enhancements

### Medium Priority
- [ ] New dashboard visualizations
- [ ] Additional data source integrations
- [ ] Internationalization (i18n)
- [ ] Accessibility improvements

### Low Priority
- [ ] UI/UX enhancements
- [ ] Documentation improvements
- [ ] Code refactoring
- [ ] Additional test coverage

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=app_enhanced

# Run specific test file
python -m pytest tests/test_app.py
```

### Writing Tests
- Place tests in the `tests/` directory
- Follow naming convention: `test_*.py`
- Use descriptive test function names
- Mock external dependencies

## 📚 Documentation

### Updating Documentation
- Update README.md for user-facing changes
- Update docstrings for code changes
- Add comments for complex logic
- Update API documentation if applicable

### Documentation Style
- Use clear, concise language
- Include code examples
- Add screenshots for UI changes
- Keep documentation up to date

## 🏗️ Architecture Guidelines

### Adding New Features
1. **Plan** the feature architecture
2. **Design** the user interface mockups
3. **Implement** in small, testable chunks
4. **Document** the feature
5. **Test** thoroughly

### Code Organization
```
app_enhanced.py          # Main application entry point
├── utils/              # Utility functions
├── models/             # Data models and AI/ML models
├── visualizations/     # Chart and visualization components
├── data/              # Data processing and loading
└── components/        # Reusable UI components
```

## 🔄 Release Process

### Versioning
- Follow [Semantic Versioning](https://semver.org/)
- Format: MAJOR.MINOR.PATCH
- Update version in relevant files

### Release Checklist
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Version bumped
- [ ] Release notes prepared

## 🎖️ Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Special thanks in major releases

## 📞 Getting Help

### Communication Channels
- **Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Email**: For private communication

### Response Times
- Issues: Within 48 hours
- Pull Requests: Within 72 hours
- Questions: Within 24 hours

## 📋 Commit Message Guidelines

### Format
```
type(scope): description

Body (optional)

Footer (optional)
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions or changes
- `chore`: Maintenance tasks

### Examples
```
feat(dashboard): add revenue forecasting model

Add AI-powered revenue forecasting with 79% accuracy
- Implement Prophet time series model
- Add confidence intervals
- Update dashboard with forecast charts

Closes #123
```

## 🙏 Thank You

Your contributions make this project better for everyone. Whether you're fixing bugs, adding features, improving documentation, or helping other users, your efforts are appreciated!

---

*This contributing guide is a living document and will be updated as the project evolves.*