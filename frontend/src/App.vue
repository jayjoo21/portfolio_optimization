<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
</script>

<template>
  <div class="app-shell">
    <aside class="sidebar">
      <div>
        <!-- BRAND -->
        <div class="brand">
          <div class="brand-mark">P</div>

          <div class="brand-text">
            <strong> Portfolio </strong>

            <span> Optimization </span>
          </div>
        </div>

        <!-- NAVIGATION -->
        <nav class="navigation">
          <p class="nav-section">ANALYSIS</p>

          <RouterLink to="/">
            <span class="nav-icon"> 01 </span>

            <span> Overview </span>
          </RouterLink>

          <RouterLink to="/replay">
            <span class="nav-icon"> 02 </span>

            <span> Decision Replay </span>
          </RouterLink>

          <RouterLink to="/backtest">
            <span class="nav-icon"> 03 </span>

            <span> Backtest </span>
          </RouterLink>

          <RouterLink to="/execution">
            <span class="nav-icon"> 04 </span>

            <span> Execution </span>
          </RouterLink>
        </nav>
      </div>

      <!-- SIDEBAR FOOTER -->
      <div class="sidebar-footer">
        <div class="prototype-tag">PROTOTYPE · MOCK DATA</div>

        <div class="model-status">
          <span class="status-dot"></span>

          <span> MODEL </span>
        </div>

        <strong> Deep RL · EIIE </strong>

        <small> Historical simulation </small>
      </div>
    </aside>

    <!-- PAGE -->
    <section class="content">
      <RouterView v-slot="{ Component }">
        <Transition name="page" mode="out-in">
          <component :is="Component" />
        </Transition>
      </RouterView>
    </section>
  </div>
</template>

<style scoped>
.app-shell {
  display: flex;

  min-height: 100vh;
}

.sidebar {
  position: sticky;

  top: 0;

  display: flex;

  width: 238px;
  height: 100vh;

  flex-shrink: 0;
  flex-direction: column;
  justify-content: space-between;

  padding: 28px 18px 22px;

  border-right: 1px solid #e7e7e9;

  background: rgba(255, 255, 255, 0.92);

  backdrop-filter: blur(16px);
}

/* BRAND */

.brand {
  display: flex;

  gap: 11px;

  align-items: center;

  padding: 0 8px;
}

.brand-mark {
  display: grid;

  width: 36px;
  height: 36px;

  flex-shrink: 0;

  place-items: center;

  border-radius: 10px;

  background: #18181b;

  color: #ffffff;

  font-size: 14px;
  font-weight: 750;

  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
}

.brand-text {
  display: flex;

  flex-direction: column;

  gap: 2px;
}

.brand-text strong {
  font-size: 14px;

  letter-spacing: -0.02em;
}

.brand-text span {
  color: #929299;

  font-size: 10px;
}

/* NAV */

.navigation {
  display: flex;

  flex-direction: column;

  gap: 4px;

  margin-top: 42px;
}

.nav-section {
  margin: 0 0 9px 12px;

  color: #adadb4;

  font-size: 9px;
  font-weight: 700;

  letter-spacing: 0.12em;
}

.navigation a {
  position: relative;

  display: flex;

  gap: 11px;

  align-items: center;

  padding: 11px 12px;

  border-radius: 9px;

  color: #73737b;

  font-size: 12px;
  font-weight: 500;

  transition:
    background 150ms ease,
    color 150ms ease,
    transform 150ms ease;
}

.navigation a:hover {
  background: #f6f6f7;

  color: #18181b;

  transform: translateX(2px);
}

.navigation a.router-link-active {
  background: #f1f1f3;

  color: #18181b;

  font-weight: 650;
}

.navigation a.router-link-active::before {
  position: absolute;

  top: 50%;
  left: 3px;

  width: 3px;
  height: 17px;

  border-radius: 999px;

  background: #18181b;

  content: '';

  transform: translateY(-50%);
}

.nav-icon {
  width: 19px;

  color: #aaaab1;

  font-size: 9px;
  font-weight: 700;
}

.navigation a.router-link-active .nav-icon {
  color: #18181b;
}

/* FOOTER */

.sidebar-footer {
  display: flex;

  flex-direction: column;

  gap: 4px;

  padding: 16px 10px 4px;

  border-top: 1px solid #eeeef0;
}

.model-status {
  display: flex;

  gap: 6px;

  align-items: center;

  margin-bottom: 3px;

  color: #a1a1aa;

  font-size: 9px;
  font-weight: 700;

  letter-spacing: 0.1em;
}

.status-dot {
  width: 6px;
  height: 6px;

  border-radius: 50%;

  background: #4b8a61;

  box-shadow: 0 0 0 3px rgba(75, 138, 97, 0.1);
}

.sidebar-footer strong {
  font-size: 12px;
}

.sidebar-footer small {
  color: #8b8b93;

  font-size: 10px;
}

.content {
  min-width: 0;

  flex: 1;
}

@media (max-width: 800px) {
  .sidebar {
    width: 76px;

    padding: 25px 11px;
  }

  .brand-text,
  .navigation a > span:last-child,
  .nav-section,
  .sidebar-footer {
    display: none;
  }

  .brand {
    justify-content: center;

    padding: 0;
  }

  .navigation a {
    justify-content: center;

    padding: 12px 8px;
  }

  .nav-icon {
    width: auto;
  }
}

.prototype-tag {
  width: fit-content;

  margin-bottom: 12px;
  padding: 5px 7px;

  border: 1px solid #e4e4e7;
  border-radius: 6px;

  background: #fafafa;
  color: #929299;

  font-size: 8px;
  font-weight: 750;

  letter-spacing: 0.09em;
}
</style>
