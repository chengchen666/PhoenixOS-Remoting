# Guidelines for contributions 

## Pass tests 

The MR/PR should pass all the unittests as well as integration tests. 

## Submit a MR/PR 

### MR/PR title 

A valid PR title should begin with one of the following prefixes (see also [here](https://github.com/commitizen/conventional-commit-types/blob/master/index.json)) : 

- `feat`: A new feature
- `fix`: A bug fix
- `doc`: Documentation only changes
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `style`: A refactoring that improves code style
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `build`: Changes that affect the build system or external dependencies (example scopes: `.config`, `.cargo`, `Cargo.toml`)
- `ci`: Changes to RisingWave CI configuration files and scripts (example scopes: `.github`, `ci` (Buildkite))
- `chore`: Other changes that don't modify src or test files
- `revert`: Reverts a previous commit

For example, a MR/PR title could be:

- `refactor: refine the communication of RDMA `
- `feat(network): add RDMA as a communication backend`, where `(network)` means that this PR mainly focuses on the network module.

###  MR/PR description

Please decribe your overall MR/PR

- If  it is small (e.g., typo fixes), you can go brief.
- If it is large and you have changed a lot, please describe it using templates 



###  MR/PR content and issues

For the MR/PR content, please note, it it is a **new feature**, please include the following:

- A high-level design (in a separate issue) of the MR/PR. 
- Files changed. 
- Unittests for the feature. 
- Necessary comments especially for the function/module descriptions. Can use AI (e.g., https://github.com/binary-husky/gpt_academic) but should check the content manually. 
- (Recommendation) You can use tools to enhance the quality of the code quality, e.g., https://github.com/binary-husky/gpt_academic. 

